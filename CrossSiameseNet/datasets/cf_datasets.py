from torch.utils.data import Dataset
import torch
import random
import numpy as np
import logging
from CrossSiameseNet.datasets.datasets_utils import load_dataset
from rdkit.Chem import AllChem

class MolDataset(Dataset):

    def __init__(self, X, y, smiles):
        """
        Attributes
        ----------
        dc_dataset: dc_Datset
            DeepChem's dataset containing:
            - X -> Molecules' circular fingerprints
            - y -> labels
            - ids -> smiles
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.smiles = smiles
        self.n_molecules = len(self.smiles)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, id0):

        # random other molecule
        id1 = random.randint(0, self.__len__() - 1)

        # molecular fingerprints
        mf0 = self.X[id0]
        mf1 = self.X[id1]

        # smiles
        smile0 = self.smiles[id0]
        smile1 = self.smiles[id1]

        # difference between targets
        target = torch.tensor(abs(self.y[id0] - self.y[id1]), dtype=torch.float32)

        return mf0, mf1, target, smile0, smile1


class MolDatasetTriplet(MolDataset):

    def __init__(self, X, y, smiles, train: bool, oversample: int = None, 
                 use_fixed_triplets: bool = False, seed_fixed_triplets: int = None,
                 training_type: str = "hard_batch_learning"):
        
        super().__init__(X, y, smiles)
        self.train = train
        self.oversample = oversample
        self.use_fixed_triplets = use_fixed_triplets
        self.seed_fixed_triplets = seed_fixed_triplets
        self.training_type = training_type

        indices_0 = (self.y == 0).nonzero()[:,0].tolist()
        indices_1 = (self.y == 1).nonzero()[:,0].tolist()

        # set stable test triplets for repeatance
        if self.use_fixed_triplets:
            self.indices_0 = indices_0
            self.indices_1 = indices_1
            self.fixed_triplets = self.__get_fixed_dataset()
        
        else:
            self.indices_0 = indices_0
            self.indices_1 = indices_1


    def __getitem__(self, id0):

        if self.train == True:
            
            anchor_mf = self.X[id0]
            anchor_label = self.y[id0].item()
            anchor_smile = self.smiles[id0]

            # dummy tensors nad lists - the batch forming role lies on BatchShaper
            if self.training_type in ["hard_batch_learning", "hard_batch_learning_only_positives", "semi_hard_negative_mining"]:
                positive_mf = torch.tensor([0])
                negative_mf = torch.tensor([0])

                positive_smile = ""
                negative_smile = ""

            else:

                if anchor_label == 1:
                    positive_index = random.choice(self.indices_1)
                    negative_index = random.choice(self.indices_0)

                else:
                    positive_index = random.choice(self.indices_0)
                    negative_index = random.choice(self.indices_1)

                positive_mf = self.X[positive_index]
                negative_mf = self.X[negative_index]

                positive_smile = self.smiles[positive_index]
                negative_smile = self.smiles[negative_index]

        else:            
            anchor_mf, positive_mf, negative_mf, anchor_label, anchor_smile, positive_smile, negative_smile = self.fixed_triplets[0][id0], \
                self.fixed_triplets[1][id0], self.fixed_triplets[2][id0], self.fixed_triplets[3][id0], self.fixed_triplets[4][id0], \
                     self.fixed_triplets[5][id0], self.fixed_triplets[6][id0]

        return anchor_mf, positive_mf, negative_mf, anchor_label, anchor_smile, positive_smile, negative_smile


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
        positive_smiles = [self.smiles[positive_index] for positive_index in positive_indices]
        negative_smiles = [self.smiles[negative_index] for negative_index in negative_indices]

        return [anchor_mf, positive_mf, negative_mf, self.y, self.smiles, positive_smiles, negative_smiles]
    

    def refresh_fixed_triplets(self, seed_fixed_triplets: int):
        self.seed_fixed_triplets = seed_fixed_triplets
        self.fixed_triplets = self.__get_fixed_dataset()


    def shuffle_data(self, batch_size):
        '''Set order of observations in such way that each batch has the same number of observations with label 1'''
        
        n_left_observations = len(self.y)
        n_batches = int(n_left_observations / batch_size) + 1
        nominal_n_1_observations = [len(el) for el in np.array_split(np.array(range(len(self.indices_1))), n_batches)]

        indices_free_1 = set(self.indices_1.copy())
        to_sample_indices_free_1 = tuple(set(self.indices_1.copy()))
        indices_free_0 = set(self.indices_0.copy())
        to_sample_indices_free_0 = tuple(set(self.indices_0.copy()))
        indices_updated = []

        for n_batch in range(n_batches):

            actual_batch_size = min(batch_size, n_left_observations)
            n_1_observations = nominal_n_1_observations[n_batch]
            n_0_observations = actual_batch_size - n_1_observations
            n_left_observations = n_left_observations - actual_batch_size

            ids0 = random.sample(to_sample_indices_free_0, n_0_observations)
            ids1 = random.sample(to_sample_indices_free_1, n_1_observations)

            # collect indices
            indices_updated = indices_updated + list(ids1) + list(ids0)

            # remove used indices 
            indices_free_0 = indices_free_0 - set(ids0)
            indices_free_1 = indices_free_1 - set(ids1)

        # update data
        self.X = self.X[indices_updated, :]
        self.y = self.y[indices_updated, :]
        self.smiles = [self.smiles[i] for i in indices_updated]

        self.indices_0 = (self.y == 0).nonzero()[:,0].tolist()
        self.indices_1 = (self.y == 1).nonzero()[:,0].tolist()


def get_dataset(dataset_name: str, splitter = None, cf_radius: int = 4, cf_size: int = 2048, 
                triplet_loss = False, oversample: int = None, use_fixed_train_triplets: bool = False, 
                seed_fixed_train_triplets: int = None, training_type: str = "hard_batch_learning",
                ogbg_dataset_path = None):
    '''Downloads DeepChem's dataset and wraprs them into a Torch dataset
    
    Available datasets:
    - HIV (inhibit HIV replication)
    - Delaney (solubility)
    - Lipo (lipophilicity)
    - Tox21
    '''

    if use_fixed_train_triplets and not triplet_loss:
        logging.warning("Fixed triplets for regular dataset not implemented yet")
        return None

    featurizer = AllChem.GetMorganGenerator(radius=cf_radius, fpsize=cf_size)
    X_train, y_train, smiles_train, X_val, y_val, smiles_val, \
        X_test, y_test, smiles_test = load_dataset(dataset_name, featurizer, splitter, ogbg_dataset_path)    

    # convert DeepChems datasets to Torch wrappers
    if triplet_loss:
        train_dataset = MolDatasetTriplet(X_train, y_train, smiles_train, True, oversample, use_fixed_train_triplets, 
                                            seed_fixed_train_triplets, training_type)
        test_dataset = MolDatasetTriplet(X_test, y_test, smiles_test, False, False, True, 123)
        valid_dataset = MolDatasetTriplet(X_val, y_val, smiles_val, False, False, True, 123)
    
    else:
        train_dataset, valid_dataset, test_dataset = \
            MolDataset(X_train, y_train, smiles_train), MolDataset(X_val, y_val, smiles_val), MolDataset(X_test, y_test, smiles_test)

    return train_dataset, valid_dataset, test_dataset