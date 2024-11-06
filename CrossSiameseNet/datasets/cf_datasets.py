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
                 model = None, device = None, training_type: str = "hard_batch_learning",
                 k_hard_batch_learning: int = 6):
        
        super().__init__(dc_dataset)
        self.train = train
        self.oversample = oversample
        self.use_fixed_triplets = use_fixed_triplets
        self.seed_fixed_triplets = seed_fixed_triplets
        self.model = model
        self.device = device
        self.training_type = training_type
        self.k_hard_batch_learning = k_hard_batch_learning
        self.euclidean_distance = torch.nn.PairwiseDistance(p=2)

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

            # random positive and negative samples
            if self.training_type == "hard_batch_learning":
                print(f"Entering training type: {self.training_type}")
                anchor_mf, positive_mf, negative_mf = self.__hard_batch_learning(anchor_label, anchor_mf, self.k_hard_batch_learning)
                already_transformed_mfs = True

            elif self.training_type == "hard_batch_learning_only_positives":
                print(f"Entering training type: {self.training_type}")
                anchor_mf, positive_mf, negative_mf = self.__hard_batch_learning_only_positives(anchor_label, anchor_mf, self.k_hard_batch_learning)
                print(f"anchor_mf.shape: {anchor_mf.shape}")
                print(f"positive_mf.shape: {positive_mf.shape}")
                print(f"negative_mf.shape: {negative_mf.shape}")

                already_transformed_mfs = True

            else:

                if anchor_label == 1:
                    positive_index = random.choice(self.indices_1)
                    negative_index = random.choice(self.indices_0)

                else:
                    positive_index = random.choice(self.indices_0)
                    negative_index = random.choice(self.indices_1)

                already_transformed_mfs = False
                positive_mf = self.X[positive_index]
                negative_mf = self.X[negative_index]

        else:            
            already_transformed_mfs = False
            anchor_mf, positive_mf, negative_mf, anchor_label = self.fixed_triplets[0][id0], self.fixed_triplets[1][id0], \
                self.fixed_triplets[2][id0], self.fixed_triplets[3][id0]

        return anchor_mf, positive_mf, negative_mf, anchor_label, already_transformed_mfs


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
    

    def __hard_batch_learning_only_positives(self, anchor_label, anchor_mf, k=6):

        # anchor_mf needs to be stacked because of the batch norm that requires n > 1 obs
        anchor_mf = torch.stack([anchor_mf for i in range(2)], dim=0).to(self.device)
        anchor_mf_transformed = self.model(anchor_mf)[0]
        print(f"anchor_label: {anchor_label}")

        # looking for toughest negative and positive samples for positive anchor
        if anchor_label == 1:
            
            print("Processing anchor label 1")
            positive_indices = random.sample(self.indices_1, k=k)
            negative_indices = random.sample(self.indices_0, k=k)
            print(f"len(positive_indices): {len(positive_indices)}")
            print(f"len(negative_indices): {len(negative_indices)}")

            positive_mfs = self.X[positive_indices]
            negative_mfs = self.X[negative_indices]
            print("BEFORE MODEL")
            print(f"positive_mfs.shape: {positive_mfs.shape}")
            print(f"negative_mfs.shape: {negative_mfs.shape}")

            positive_mfs = self.model(positive_mfs.to(self.device))
            negative_mfs = self.model(negative_mfs.to(self.device))
            print(f"positive_mfs.shape: {positive_mfs.shape}")
            print(f"negative_mfs.shape: {negative_mfs.shape}")
            print("AFTER MODEL")

            # find toughest positive and negative observation
            min_dist = -1 
            for index in range(k):
                positive_mf = positive_mfs[index]
                dist = self.euclidean_distance(anchor_mf_transformed, positive_mf).item()
                if dist > min_dist:
                    print(f"min_dist: {min_dist} --> dist: {dist}")
                    min_dist = dist
                    positive_index = index
                print(f"min_dist: {min_dist}")
                print(f"positive_index: {positive_index}")

            max_dist = float("inf") 
            for index in range(k):
                negative_mf = negative_mfs[index]
                dist = self.euclidean_distance(anchor_mf_transformed, negative_mf).item()
                if dist < max_dist:
                    print(f"max_dist: {max_dist} --> dist: {dist}")
                    max_dist = dist
                    negative_index = index
                print(f"max_dist: {max_dist}")
                print(f"negative_index: {negative_index}")

            positive_mf = positive_mfs[positive_index]
            negative_mf = negative_mfs[negative_index]
            print(f"positive_mf.shape: {positive_mf.shape}")
            print(f"negative_mf.shape: {negative_mf.shape}")

        # random positive and negative observations
        else:

            positive_index = random.choice(self.indices_0)
            negative_index = random.choice(self.indices_1)

            positive_mf = self.X[positive_index]
            negative_mf = self.X[negative_index]

            positive_mf = torch.stack([positive_mf for i in range(2)], dim=0).to(self.device)
            negative_mf = torch.stack([negative_mf for i in range(2)], dim=0).to(self.device)

            positive_mf = self.model(positive_mf)[0]
            negative_mf = self.model(negative_mf)[0]

        # return positive_index, negative_index
        return anchor_mf_transformed, positive_mf, negative_mf


    def __hard_batch_learning(self, anchor_label, anchor_mf, k=6):

        # anchor_mf needs to be stacked because of the batch norm that requires n > 1 obs
        anchor_mf = torch.stack([anchor_mf for i in range(2)], dim=0).to(self.device)
        anchor_mf_transformed = self.model(anchor_mf)[0]

        if anchor_label == 1:
            positive_indices = random.sample(self.indices_1, k=k)
            negative_indices = random.sample(self.indices_0, k=k)
        
        else:
            positive_indices = random.sample(self.indices_0, k=k)
            negative_indices = random.sample(self.indices_1, k=k)

        positive_mfs = self.X[positive_indices]
        negative_mfs = self.X[negative_indices]

        positive_mfs = self.model(positive_mfs.to(self.device))
        negative_mfs = self.model(negative_mfs.to(self.device))

        # find toughest positive and negative observation
        min_dist = -1 
        for index in range(k):
            positive_mf = positive_mfs[index]
            dist = self.euclidean_distance(anchor_mf_transformed, positive_mf).item()
            if dist > min_dist:
                min_dist = dist
                positive_index = index

        max_dist = float("inf") 
        for index in range(k):
            negative_mf = negative_mfs[index]
            dist = self.euclidean_distance(anchor_mf_transformed, negative_mf).item()
            if dist < max_dist:
                max_dist = dist
                negative_index = index

        # return positive_index, negative_index
        return anchor_mf_transformed, positive_mfs[positive_index], negative_mfs[negative_index]


    def refresh_fixed_triplets(self, seed_fixed_triplets: int):
        self.seed_fixed_triplets = seed_fixed_triplets
        self.fixed_triplets = self.__get_fixed_dataset()


def get_dataset(dataset_name: str, splitter: Splitter = None, cf_radius: int = 4, cf_size: int = 2048, 
                triplet_loss = False, oversample: int = None, use_fixed_train_triplets: bool = False, 
                seed_fixed_train_triplets: int = None, training_type: str = "hard_batch_learning", k_hard_batch_learning: int = 6):
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
            train_dataset = MolDatasetTriplet(datasets[0], True, oversample, use_fixed_train_triplets, seed_fixed_train_triplets, 
                                              training_type = training_type, k_hard_batch_learning = k_hard_batch_learning)
            valid_dataset = MolDatasetTriplet(datasets[1], False, False, True, 123)
            test_dataset = MolDatasetTriplet(datasets[2], False, False, True, 123)
        
        else:
            train_dataset, valid_dataset, test_dataset = \
                MolDataset(datasets[0]), MolDataset(datasets[1]), MolDataset(datasets[2])

        return train_dataset, valid_dataset, test_dataset

    # dataset wrapped in one object
    else:

        return datasets[0]