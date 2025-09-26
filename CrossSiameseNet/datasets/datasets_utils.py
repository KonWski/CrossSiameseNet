from deepchem.molnet import load_hiv, load_delaney, load_lipo, load_tox21
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors
import numpy as np

def load_dataset(dataset_name, featurizer, splitter):
    
    if dataset_name == "hiv":
        _, datasets, _ = load_hiv(featurizer, splitter)

    elif dataset_name == "hiv_esol":
        datasets = load_hiv_esol(featurizer, splitter)

    elif dataset_name == "delaney":
        _, datasets, _ = load_delaney(featurizer, splitter)
    
    elif dataset_name == "lipo":
        _, datasets, _ = load_lipo(featurizer, splitter)
    
    elif dataset_name[:5] == "tox21":
        task = dataset_name[dataset_name.find("_")+1:]
        _, datasets, _ = load_tox21(featurizer, splitter, tasks=[task])
    
    else:
        raise Exception(f"Dataset {dataset_name} not implemented")

    train_dataset, val_dataset, test_dataset = datasets
    X_train, y_train, smiles_train = train_dataset.X, train_dataset.y, train_dataset.smiles
    X_val, y_val, smiles_val = val_dataset.X, val_dataset.y, val_dataset.smiles
    X_test, y_test, smiles_test = test_dataset.X, test_dataset.y, test_dataset.smiles

    return X_train, y_train, smiles_train, X_val, y_val, smiles_val, X_test, y_test, smiles_test


def load_hiv_esol(featurizer, splitter):
    _, datasets, _ = load_hiv(featurizer, splitter)
    train_dataset, val_dataset, test_dataset = datasets

    X_train, smiles_train = train_dataset.X, train_dataset.ids
    X_val, smiles_val = val_dataset.X, val_dataset.ids
    X_test, smiles_test = test_dataset.X, test_dataset.ids

    y_train = np.array([esol_predict(smiles) for smiles in smiles_train])
    y_val = np.array([esol_predict(smiles) for smiles in smiles_val])
    y_test = np.array([esol_predict(smiles) for smiles in smiles_test])
    
    return DummyDataset(X_train, y_train, smiles_train), \
        DummyDataset(X_val, y_val, smiles_val), \
        DummyDataset(X_test, y_test, smiles_test)
            

def esol_predict(smiles: str):

    mol = Chem.MolFromSmiles(smiles)

    # octanol-water partition coefficient, molecular weight, rotatable bonds 
    logP = Crippen.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    aromatic_proportion = (
        sum(1 for a in mol.GetAromaticAtoms()) / mol.GetNumAtoms()
        if mol.GetNumAtoms() > 0 else 0
    )

    esol_solubility = 0.16 - 0.63 * logP - 0.0062 * mw + 0.066 * rot_bonds - 0.74 * aromatic_proportion

    return esol_solubility

class DummyDataset():

    def __init__(self, X, y, smiles):
        self.X = X
        self.y = y
        self.smiles = smiles