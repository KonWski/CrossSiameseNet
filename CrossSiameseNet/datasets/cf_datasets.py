from torch.utils.data import Dataset
from deepchem.data.datasets import Dataset as dc_Datset
from deepchem.molnet import load_hiv, load_delaney, load_lipo, load_freesolv, load_tox21
from deepchem.splits.splitters import Splitter
from deepchem.feat import CircularFingerprint
import torch
import random
import numpy as np
import logging

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

    def __init__(self, dc_dataset: dc_Datset, train: bool, use_fixed_triplets: bool = False, seed_fixed_triplets: int = None):
        
        super().__init__(dc_dataset)
        self.train = train

        indices_0_temp = self.y == 0
        self.indices_0 = indices_0_temp.nonzero()[:,0].tolist()

        indices_1_temp = self.y == 1
        self.indices_1 = indices_1_temp.nonzero()[:,0].tolist()

        self.use_fixed_triplets = use_fixed_triplets
        self.seed_fixed_triplets = seed_fixed_triplets

        # set stable test triplets for repeatance
        if self.use_fixed_triplets:
            self.fixed_triplets = self.__get_fixed_dataset()


    def __getitem__(self, id0):

        if not self.use_fixed_triplets and self.train == False:
            
            anchor_mf = self.X[id0]
            anchor_label = self.y[id0].item()

            # random positive and negative samples
            if anchor_label == 1:
                positive_index = random.choice(self.indices_1)
                negative_index = random.choice(self.indices_0)
            
            else:
                positive_index = random.choice(self.indices_0)
                negative_index = random.choice(self.indices_1)

            positive_mf = self.X[positive_index]
            negative_mf = self.X[negative_index]

        # oversampling
        elif not self.use_fixed_triplets and self.train == True:
            
            # switch selected id
            anchor_label = np.random.choice(2, 1, p=[0.5, 0.5]).item()

            # random positive and negative samples
            if anchor_label == 1:
                id0 = random.choice(self.indices_1)
                anchor_mf = self.X[id0]
                positive_index = random.choice(self.indices_1)
                negative_index = random.choice(self.indices_0)
            
            else:
                id0 = random.choice(self.indices_0)
                anchor_mf = self.X[id0]
                positive_index = random.choice(self.indices_0)
                negative_index = random.choice(self.indices_1)

            positive_mf = self.X[positive_index]
            negative_mf = self.X[negative_index]


        else:            
            anchor_mf, positive_mf, negative_mf, anchor_label = self.fixed_triplets[0][id0], self.fixed_triplets[1][id0], \
                self.fixed_triplets[2][id0], self.fixed_triplets[3][id0]

        return anchor_mf, positive_mf, negative_mf, anchor_label


    def __get_fixed_dataset(self):

        random_state = np.random.RandomState(self.seed_fixed_triplets)

        # random positive and negative samples
        anchor_mf = self.X
        positive_indices = []
        negative_indices = []

        for label_packed in self.y.tolist():
            
            anchor_label = label_packed[0]

            if anchor_label == 1:
                positive_indices.append(random_state.choice(self.indices_1))
                negative_indices.append(random_state.choice(self.indices_0))

            else:
                positive_indices.append(random_state.choice(self.indices_0))
                negative_indices.append(random_state.choice(self.indices_1))

        positive_mf = self.X[positive_indices]
        negative_mf = self.X[negative_indices]

        return [anchor_mf, positive_mf, negative_mf, anchor_label]
    

    def get_pos_weights(self):
        return torch.tensor([1, len(self.indices_0) / len(self.indices_1)])


    def refresh_fixed_triplets(self, seed_fixed_triplets: int):
        self.seed_fixed_triplets = seed_fixed_triplets
        self.fixed_triplets = self.__get_fixed_dataset()


def get_dataset(dataset_name: str, splitter: Splitter = None, cf_radius: int = 4, cf_size: int = 2048, 
                triplet_loss = False, use_fixed_train_triplets: bool = False, seed_fixed_train_triplets: int = None):
    '''Downloads DeepChem's dataset and wraprs them into a Torch dataset
    
    Available datasets:
    - HIV (inhibit HIV replication)
    - Delaney (solubility)
    - Lipo (lipophilicity)
    - FreeSolv (octanol/water distribution)
    - Tox21
    '''

    if use_fixed_train_triplets and not triplet_loss:
        logging.warning("Fixed triplets for regular dataset not implemented yet")
        return None

    featurizer = CircularFingerprint(cf_radius, cf_size)

    if dataset_name == "hiv":
        _, datasets, _ = load_hiv(featurizer, splitter)

    elif dataset_name == "delaney":
        _, datasets, _ = load_delaney(featurizer, splitter)
    
    elif dataset_name == "lipo":
        _, datasets, _ = load_lipo(featurizer, splitter)
    
    elif dataset_name == "freesolv":
        _, datasets, _ = load_freesolv(featurizer, splitter)
    
    elif dataset_name[:5] == "tox21":
        task = dataset_name[dataset_name.find("_")+1:]
        _, datasets, _ = load_tox21(featurizer, splitter, tasks=[task])

    if splitter is not None:

        # convert DeepChems datasets to Torch wrappers
        if triplet_loss:
            train_dataset = MolDatasetTriplet(datasets[0], True, use_fixed_train_triplets, seed_fixed_train_triplets)
            valid_dataset = MolDatasetTriplet(datasets[1], False, True, 123)
            test_dataset = MolDatasetTriplet(datasets[2], False, True, 123)
        
        else:
            train_dataset, valid_dataset, test_dataset = \
                MolDataset(datasets[0]), MolDataset(datasets[1]), MolDataset(datasets[2])

        return train_dataset, valid_dataset, test_dataset

    # dataset wrapped in one object
    else:

        return datasets[0]