from torch.utils.data import Dataset
from deepchem.data.datasets import Dataset as dc_Datset
from deepchem.molnet import load_hiv, load_delaney, load_lipo, load_freesolv, load_tox21
from deepchem.splits.splitters import Splitter
from deepchem.feat import CircularFingerprint
import torch
import random
import numpy as np

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


class MolDatasetTriplet(MolDataset):

    def __init__(self, dc_dataset: dc_Datset, train: bool):
        
        super().__init__(dc_dataset)
        self.train = train

        indices_0_temp = self.y == 0
        self.indices_0 = indices_0_temp.nonzero()[:,0].tolist()

        indices_1_temp = self.y == 1
        self.indices_1 = indices_1_temp.nonzero()[:,0].tolist()

        # set stable test triplets for repeatance
        if not self.train:
            random_state = np.random.RandomState(123)

            self.test_triplets = []

            # random positive and negative samples
            anchor_mf = self.X
            positive_indices = []
            negative_indices = []

            for label in self.y.tolist():
                
                if label == 1:
                    positive_indices.append(random_state.choice(self.indices_0))
                    negative_indices.append(random_state.choice(self.indices_1))

                else:
                    positive_indices.append(random_state.choice(self.indices_1))
                    negative_indices.append(random_state.choice(self.indices_0))

            positive_mf = self.X[positive_indices]
            negative_mf = self.X[negative_indices]
            self.test_triplets = [anchor_mf, positive_mf, negative_mf]

    def __getitem__(self, id0):

        if self.train:
            
            anchor_mf = self.X[id0]
            anchor_labels = self.y[id0]

            # random positive and negative samples
            positive_indices = []
            negative_indices = []

            for label in anchor_labels.tolist():
                if label == 1:
                    positive_indices.append(random.choice(self.indices_0))
                    negative_indices.append(random.choice(self.indices_1))

                else:
                    positive_indices.append(random.choice(self.indices_1))
                    negative_indices.append(random.choice(self.indices_0))

            positive_mf = self.X[positive_indices]
            negative_mf = self.X[negative_indices]

        else:
            
            anchor_mf, positive_mf, negative_mf = self.test_triplets[0][id0], self.test_triplets[1][id0], self.test_triplets[2][id0]

        return anchor_mf, positive_mf, negative_mf


def get_dataset(dataset_name: str, splitter: Splitter = None, cf_radius=4, cf_size=2048):
    '''Downloads DeepChem's dataset and wraprs them into a Torch dataset
    
    Available datasets:
    - HIV (inhibit HIV replication)
    - Delaney (solubility)
    - Lipo (lipophilicity)
    - FreeSolv (octanol/water distribution)
    '''

    featurizer = CircularFingerprint(cf_radius, cf_size)

    if dataset_name == "hiv":
        _, datasets, _ = load_hiv(featurizer, splitter)

    elif dataset_name == "delaney":
        _, datasets, _ = load_delaney(featurizer, splitter)
    
    elif dataset_name == "lipo":
        _, datasets, _ = load_lipo(featurizer, splitter)
    
    elif dataset_name == "freesolv":
        _, datasets, _ = load_freesolv(featurizer, splitter)
    
    elif dataset_name == "tox21":
        _, datasets, _ = load_tox21(featurizer, splitter, tasks=['NR-AR'])

    if splitter is not None:

        # convert DeepChems datasets to Torch wrappers
        train_dataset, valid_dataset, test_dataset = \
            MolDatasetTriplet(datasets[0], True), MolDatasetTriplet(datasets[1], False), MolDatasetTriplet(datasets[2], False)
        return train_dataset, valid_dataset, test_dataset

    # dataset wrapped in one object
    else:
        return datasets[0]