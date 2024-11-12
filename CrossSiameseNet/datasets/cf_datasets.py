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
        return len(self.y)

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

    def __init__(self, dc_dataset: dc_Datset, train: bool, oversample: int = None, 
                 use_fixed_triplets: bool = False, seed_fixed_triplets: int = None,
                 training_type: str = "hard_batch_learning"):
        
        super().__init__(dc_dataset)
        self.train = train
        self.oversample = oversample
        self.use_fixed_triplets = use_fixed_triplets
        self.seed_fixed_triplets = seed_fixed_triplets
        self.training_type = training_type

        indices_0 = (self.y == 0).nonzero()[:,0].tolist()
        indices_1 = (self.y == 1).nonzero()[:,0].tolist()

        if self.oversample and not self.use_fixed_triplets:

            X_0 = self.X[indices_0]
            X_1 = self.X[indices_1]
            y_0 = self.y[indices_0]
            y_1 = self.y[indices_1]
            smiles_0 = self.smiles[indices_0]
            smiles_1 = self.smiles[indices_1]

            oversampled_X_1 = torch.cat([X_1 for i in range(self.oversample)])
            oversampled_y_1 = torch.cat([y_1 for i in range(self.oversample)])
            oversampled_smiles_1 = np.concatenate([smiles_1 for i in range(self.oversample)])

            self.X = torch.cat([X_0, oversampled_X_1])
            self.y = torch.cat([y_0, oversampled_y_1])
            self.smiles = np.concatenate([smiles_0, oversampled_smiles_1])

            self.indices_0 = (self.y == 0).nonzero()[:,0].tolist()
            self.indices_1 = (self.y == 1).nonzero()[:,0].tolist()
            
        elif not self.oversample and not self.use_fixed_triplets:
            
            self.indices_0 = indices_0
            self.indices_1 = indices_1

        elif self.oversample and self.use_fixed_triplets:            
            raise Exception(f"MolDatasetTriplet initiated with wrong parameters: oversample(value: {self.oversample}) and use_fixed_triplets(value: {self.use_fixed_triplets})")

        # set stable test triplets for repeatance
        elif self.use_fixed_triplets:
            
            self.indices_0 = indices_0
            self.indices_1 = indices_1
            self.fixed_triplets = self.__get_fixed_dataset()


    def __getitem__(self, id0):

        if self.train == True:
            
            anchor_mf = self.X[id0]
            anchor_label = self.y[id0].item()

            # dummy tensors
            if self.training_type in ["hard_batch_learning", "hard_batch_learning_only_positives", "semi_hard_negative_mining"]:
                positive_mf = torch.tensor([0])
                negative_mf = torch.tensor([0])

            else:

                if anchor_label == 1:
                    positive_index = random.choice(self.indices_1)
                    negative_index = random.choice(self.indices_0)

                else:
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

        return [anchor_mf, positive_mf, negative_mf, self.y]
    

    def refresh_fixed_triplets(self, seed_fixed_triplets: int):
        self.seed_fixed_triplets = seed_fixed_triplets
        self.fixed_triplets = self.__get_fixed_dataset()


    def shuffle_data(self, batch_size):
        '''Set order of observations in such way that each batch has the same number of observations with label 1'''
        
        n_left_observations = len(self.y)
        # print(f"n_left_observations: {n_left_observations}")
        # prop_1_to_rest = len(self.indices_1) / n_left_observations
        # print(f"prop_1_to_rest: {prop_1_to_rest}")
        n_batches = int(n_left_observations / batch_size) + 1
        # print(f"n_batches: {n_batches}")
        # nominal_n_1_observations = int(prop_1_to_rest * batch_size)
        nominal_n_1_observations = [len(el) for el in np.array_split(np.array(range(len(self.indices_1))), n_batches)]

        # print(f"nominal_n_1_observations: {nominal_n_1_observations}")

        indices_free_1 = set(self.indices_1.copy())
        indices_free_0 = set(self.indices_0.copy())
        indices_updated = []

        for n_batch in range(n_batches):

            # print(f"n_batch: {n_batch}")            
            actual_batch_size = min(batch_size, n_left_observations)
            # print(f"actual_batch_size: {actual_batch_size}")
            # n_1_observations = min(nominal_n_1_observations, len(indices_free_1))
            n_1_observations = nominal_n_1_observations[n_batch]
            # print(f"n_1_observations: {n_1_observations}")
            n_0_observations = actual_batch_size - n_1_observations
            # print(f"n_0_observations: {n_0_observations}")
            n_left_observations = n_left_observations - actual_batch_size
            # print(f"n_left_observations: {n_left_observations}")

            # select random indices for updated dataset
            ids0 = random.sample(indices_free_0, n_0_observations)
            ids1 = random.sample(indices_free_1, n_1_observations)

            # collect indices
            indices_updated = indices_updated + list(ids1) + list(ids0)

            # remove used indices 
            indices_free_0 = indices_free_0 - set(ids0)
            # print(f"indices_free_0: {len(indices_free_0)}")
            indices_free_1 = indices_free_1 - set(ids1)
            # print(f"indices_free_1: {len(indices_free_1)}")

        # update data
        self.X = self.X[indices_updated, :]
        self.y = self.y[indices_updated, :]
        self.smiles = [self.smiles[i] for i in indices_updated]

        self.indices_0 = (self.y == 0).nonzero()[:,0].tolist()
        self.indices_1 = (self.y == 1).nonzero()[:,0].tolist()


def get_dataset(dataset_name: str, splitter: Splitter = None, cf_radius: int = 4, cf_size: int = 2048, 
                triplet_loss = False, oversample: int = None, use_fixed_train_triplets: bool = False, 
                seed_fixed_train_triplets: int = None, training_type: str = "hard_batch_learning"):
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
            train_dataset = MolDatasetTriplet(datasets[0], True, oversample, use_fixed_train_triplets, 
                                              seed_fixed_train_triplets, training_type)
            valid_dataset = MolDatasetTriplet(datasets[1], False, False, True, 123)
            test_dataset = MolDatasetTriplet(datasets[2], False, False, True, 123)
        
        else:
            train_dataset, valid_dataset, test_dataset = \
                MolDataset(datasets[0]), MolDataset(datasets[1]), MolDataset(datasets[2])

        return train_dataset, valid_dataset, test_dataset

    # dataset wrapped in one object
    else:

        return datasets[0]