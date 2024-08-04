import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
import logging
from datetime import datetime
import pandas as pd
from typing import List
from CrossSiameseNet.SiameseMolNet import SiameseMolNet
from CrossSiameseNet.SiameseMolNetTriplet import SiameseMolNetTriplet


class CrossSiameseNet(nn.Module):
    '''Siamese network using features from other siamese networks'''

    def __init__(self, models: List[nn.Module]):

        super().__init__()

        self.models = models
        self.n_models = len(models)
        self.cf_size = models[0].cf_size

        self.features = nn.Sequential(
            nn.Conv1d(self.n_models, 1, 1),
            nn.ReLU(inplace=True),
            nn.Flatten(start_dim=1),
            nn.BatchNorm1d(2*self.cf_size),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(2*self.cf_size, self.cf_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(self.cf_size),
            nn.Dropout(p=0.2, inplace=True)
        ) 

        self.features = nn.Sequential(
            nn.Conv1d(self.n_models, 2*self.n_models, 1),
            nn.ReLU(inplace=True),
            nn.Flatten(start_dim=1),
            nn.BatchNorm1d(2*self.cf_size * 2 * self.n_models),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(2*self.cf_size * 2 * self.n_models, self.cf_size * self.n_models),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(self.cf_size * self.n_models),
            nn.Dropout(p=0.2, inplace=True)
        ) 

        self.features = nn.Sequential(
            nn.Conv1d(self.n_models, self.n_models, 1),
            nn.ReLU(inplace=True),
            nn.Flatten(start_dim=1),
            nn.BatchNorm1d(2*self.cf_size * self.n_models),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(2*self.cf_size * self.n_models, 2* self.cf_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(2* self.cf_size),
            nn.Dropout(p=0.2, inplace=True)
        )

        self.features = nn.Sequential(
            nn.Conv1d(self.n_models, 1, 1),
            nn.ReLU(inplace=True),
            nn.Flatten(start_dim=1),
            nn.BatchNorm1d(2*self.cf_size),
            nn.Dropout(p=0.2, inplace=True)
        ) 

        self.features = nn.Sequential(
            nn.Conv1d(self.n_models, self.n_models, 1),
            nn.ReLU(inplace=True),
            nn.Flatten(start_dim=1),
            nn.BatchNorm1d(2*self.cf_size * self.n_models),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(2*self.cf_size * self.n_models, self.cf_size * self.n_models),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(self.cf_size*self.n_models),
            nn.Dropout(p=0.2, inplace=True)
                )
        self.features = nn.Sequential(
            nn.Conv1d(self.n_models, 1, 1),
            nn.ReLU(inplace=True),
            nn.Flatten(start_dim=1),
            nn.BatchNorm1d(2*self.cf_size),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(2*self.cf_size, self.cf_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(self.cf_size),
            nn.Dropout(p=0.2, inplace=True)
        ) 

        self.features = nn.Sequential(
            nn.Conv1d(self.n_models, 32, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),
            nn.Dropout(p=0.2, inplace=True),
            nn.Conv1d(32, 1, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1),
            nn.Dropout(p=0.2, inplace=True),
            nn.Flatten(start_dim=1)
        )

        self.features = nn.Sequential(
            nn.Conv1d(self.n_models, 16, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(16),
            nn.Dropout(p=0.2, inplace=True),
            nn.Conv1d(32, 1, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1),
            nn.Dropout(p=0.2, inplace=True),
            nn.Flatten(start_dim=1)
        ) 

        self.fc = nn.Sequential(
            nn.Linear(2*self.cf_size, 1),
            nn.Sigmoid()
        ) 

        # turn off grads in all parameters 
        for model in self.models:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

        # initialize the weights
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0.01)
        
        for layer in self.features:
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv1d):
                torch.nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0.01)

    def forward(self, x):

        # features collected across all models
        features_submodels = [model.forward_once(x) if isinstance(model, SiameseMolNet) else model.forward(x) for model in self.models]
        # print(f"features_submodels[0].shape: {features_submodels[0].shape}")
        # print(f"features_submodels[1].shape: {features_submodels[1].shape}")
        # print(f"features_submodels[2].shape: {features_submodels[1].shape}")


        features_submodels = torch.stack(features_submodels, dim=-2)
        # print(f"features_submodels.shape: {features_submodels.shape}")

        # print(f"features_submodels.shape: {features_submodels.shape}")
        features = self.features(features_submodels)
        # print(f"features.shape: {features.shape}")
        
        return features
