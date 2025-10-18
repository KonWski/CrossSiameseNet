from rdkit.Chem import AllChem
import random
import numpy as np
import logging
from rdkit import Chem
from rdkit.Chem.BRICS import BRICSDecompose
import torch
import copy

class MoleculeAugmentator:

    def __init__(
                self,
                mask_atoms: bool = False, 
                max_n_mask_atoms: int = 0, 
                prob_mask_atoms: float = 0.0,

                add_gaussian_noise: bool = False,
                prob_add_gaussian_noise: float = 0.0,
                sigma_gaussian_noise: float = 0.0,

                substructural_removal: bool = False,
                prob_substructural_removal: float = 0.0,

                fpgen: AllChem.GetMorganGenerator = None
            ):

        self.fpgen = fpgen
        self.mask_atom = mask_atoms
        self.prob_mask_atom = prob_mask_atoms
        self.max_n_mask_atom = max_n_mask_atoms
        self.add_gaussian_noise = add_gaussian_noise
        self.prob_add_gaussian_noise = prob_add_gaussian_noise
        self.gaussian_noise_sigma = sigma_gaussian_noise
        self.substructural_removal = substructural_removal
        self.prob_substructural_removal = prob_substructural_removal

        self.__sanitize_augmentator()

    def transform(self, mf, smile):

        mol = Chem.MolFromSmiles(smile)
        rwmol = Chem.RWMol(mol)
        mf_dtype = mf.dtype  

        if self.mask_atom:
            rwmol, mf = self.__mask_atom(rwmol, mf)
        
        if self.add_gaussian_noise:
            rwmol, mf = self.__add_gaussian_noise(rwmol, mf)

        if self.substructural_removal:
            rwmol, mf = self.__substructural_removal(rwmol, mf)

        mf = torch.tensor(mf, dtype=mf_dtype)

        return mf

    def transform_batch(self, mfs, smiles):

        mfs_transformed = torch.zeros(mfs.shape)
        n_mfs = mfs.shape[0]

        for mf_id in range(n_mfs):
            mfs_transformed[mf_id, :] = self.transform(mfs[mf_id], smiles[mf_id])

        return mfs

    def __mask_atom(self, rwmol, mf):

        n_atoms = rwmol.GetNumAtoms()

        # skip masking if the prob is lower or 
        # the number of atoms to mask is the same as number of molecule's atoms
        if np.random.uniform(0, 1) > self.prob_mask_atom or n_atoms == self.max_n_mask_atom:
            return rwmol, mf

        rwmol_augmented = copy.deepcopy(rwmol)

        n_mask_atoms = random.randint(1, self.max_n_mask_atom) 
        masked_atom_ids = random.sample(range(0, n_atoms), n_mask_atoms)

        for atom_id in masked_atom_ids:
            rwmol_augmented.GetAtomWithIdx(atom_id).SetAtomicNum(0)

        try:
            Chem.SanitizeMol(rwmol_augmented)
        except Chem.KekulizeException:
            return rwmol, mf

        mf = self.fpgen.GetFingerprintAsNumPy(rwmol_augmented)

        return rwmol_augmented, mf
    
    def __add_gaussian_noise(self, rwmol, mf):

        if np.random.uniform(0, 1) <= self.prob_add_gaussian_noise:
            mf = mf + np.random.normal(0, self.gaussian_noise_sigma, size=mf.shape[0])

        return rwmol, mf

    def __substructural_removal(self, rwmol, mf):

        if np.random.uniform(0, 1) <= self.prob_substructural_removal:

            substructures_smiles = list(BRICSDecompose(rwmol))
            substructure_smile = random.choice(substructures_smiles)

            mol = Chem.MolFromSmiles(substructure_smile)
            rwmol = Chem.RWMol(mol)        
            Chem.SanitizeMol(rwmol)
            mf = self.fpgen.GetFingerprintAsNumPy(rwmol)

        return rwmol, mf

    def __sanitize_augmentator(self):

        if self.mask_atom and (self.max_n_mask_atom != 0 or self.prob_mask_atom != 0.0):
            logging.info(f"Using masking atoms with {self.max_n_mask_atom} max_n_mask_atom and {self.prob_mask_atom} prob_mask_atom")
        elif self.mask_atom and (self.max_n_mask_atom == 0 or self.prob_mask_atom == 0.0):
            raise Exception("Masking atoms used with wrong parameters")

        if self.add_gaussian_noise and (self.gaussian_noise_sigma != 0 or self.prob_add_gaussian_noise != 0.0):
            logging.info(f"Adding gaussian noise with {self.gaussian_noise_sigma} gaussian_noise_sigma and {self.prob_add_gaussian_noise} prob_add_gaussian_noise")
        elif self.add_gaussian_noise and (self.gaussian_noise_sigma != 0 or self.prob_add_gaussian_noise != 0.0):
            raise Exception("Adding gaussian noise used with wrong parameters")

        if self.substructural_removal and self.prob_substructural_removal != 0.0:
            logging.info(f"Using substructural removal with {self.prob_substructural_removal} prob_substructural_removal")
        elif self.substructural_removal and self.prob_substructural_removal == 0.0:
            raise Exception("Substructural removal used with wrong parameters")