from torch.utils.data import Dataset
from deepchem.data.datasets import Dataset as dc_Datset
from deepchem.molnet import load_delaney
from deepchem.splits.splitters import Splitter
from deepchem.feat import CircularFingerprint
import torch
import random

class MolDataset(Dataset):

    def __init__(self, dc_dataset: dc_Datset):
        """
        Attributes
        ----------
        dc_dataset: dc_Datset
            DeepChem's dataset containing:
            - X -> Molecules' circular fingerprints
            - y -> labels
            - ids -> smiles
        """
        self.X = torch.from_numpy(dc_dataset.X).float()
        self.y = torch.from_numpy(dc_dataset.y).float()
        self.smiles = dc_dataset.ids
        self.n_molecules = len(self.smiles)

    def __len__(self):
        return self.n_molecules

    def __getitem__(self, id0):

        # random other molecule
        id1 = random.randint(0, self.__len__() - 1)

        # molecular fingerprints
        mf0 = self.X[id0]
        mf1 = self.X[id1]

        # difference between targets
        target = torch.tensor(abs(self.y[id0] - self.y[id1]), dtype=torch.float32)

        return mf0, mf1, target


def get_dataset(dataset_name: str, splitter: Splitter, cf_radius=4, cf_size=2048):
    '''Downloads DeepChem's dataset and wraprs them into a Torch dataset'''

    featurizer = CircularFingerprint(cf_radius, cf_size)

    if dataset_name == "delaney":
        tasks, datasets, transformers = load_delaney(featurizer, splitter)
    
    # convert DeepChems datasets to Torch wrappers
    train_dataset, valid_dataset, test_dataset = MolDataset(datasets[0]), MolDataset(datasets[1]), MolDataset(datasets[2])

    return train_dataset, valid_dataset, test_dataset