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


class ConvBlock(nn.Module):

    def __init__(self, dim_in: int, dim_out: int):

        super().__init__()
        self.conv = nn.Conv1d(dim_in, dim_out, 1)
        self.activation_function = nn.ReLU(inplace=True)
        self.batch_norm = nn.BatchNorm1d(dim_out)
        self.dropout = nn.Dropout(p=0.2, inplace=True)

    def forward(self, x):

        x = self.conv(x)
        x = self.activation_function(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        return x


class LinearBlock(nn.Module):

    def __init__(self, dim_in: int, dim_out: int):

        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.activation_function = nn.ReLU(inplace=True)
        self.batch_norm = nn.BatchNorm1d(dim_out)
        self.dropout = nn.Dropout(p=0.2, inplace=True)

    def forward(self, x, residual = None):

        if residual:
            x += residual

        x = self.linear(x)
        x = self.activation_function(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        return x
    

class CrossSiameseNet(nn.Module):
    '''Siamese network using features from other siamese networks'''

    def __init__(self, models: List[nn.Module]):

        super().__init__()

        self.models = models
        self.n_models = len(models)
        self.cf_size = models[0].cf_size

        self.conv_block1 = ConvBlock(self.n_models, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 128)
        self.conv_block4 = ConvBlock(128, 64)
        self.conv_block5 = ConvBlock(64, 1)

        self.linear_block = LinearBlock(2*self.cf_size, 2*self.cf_size)

        # turn off grads in all parameters 
        for model in self.models:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

        # initialize the weights
        for conv_block in [self.conv_block1, self.conv_block2, self.conv_block3, self.conv_block4, self.conv_block5]:
            torch.nn.init.xavier_uniform_(conv_block.conv.weight)
            conv_block.conv.bias.data.fill_(0.01)

        torch.nn.init.xavier_uniform_(self.linear_block.linear.weight)
        self.linear_block.linear.bias.data.fill_(0.01)

    def forward(self, x):

        # features collected across all models
        features_submodels = [model.forward_once(x) if isinstance(model, SiameseMolNet) else model.forward(x) for model in self.models]
        # print(f"features_submodels[0].shape: {features_submodels[0].shape}")
        # print(f"features_submodels[1].shape: {features_submodels[1].shape}")
        # print(f"features_submodels[2].shape: {features_submodels[1].shape}")


        features_submodels = torch.stack(features_submodels, dim=-2)
        # print(f"features_submodels.shape: {features_submodels.shape}")

        # print(f"features_submodels.shape: {features_submodels.shape}")
        x = self.conv_block1(features_submodels)
        residual_features = x
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x, residual_features)
        x = self.linear_block(x)
        
        # features = self.features(features_submodels)
        # print(f"features.shape: {features.shape}")
        
        return x