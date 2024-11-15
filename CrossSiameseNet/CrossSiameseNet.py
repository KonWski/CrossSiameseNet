import torch
import torch.nn as nn
from typing import List


class ConvBlock(nn.Module):

    def __init__(self, dim_in: int, dim_out: int):

        super().__init__()
        self.conv = nn.Conv1d(dim_in, dim_out, 1)
        self.activation_function = nn.ReLU(inplace=True)
        self.batch_norm = nn.BatchNorm1d(dim_out)

    def forward(self, x, residual = None):

        if residual is not None:
            x += residual

        x = self.conv(x)
        x = self.activation_function(x)
        x = self.batch_norm(x)
        return x


class LinearBlock(nn.Module):

    def __init__(self, dim_in: int, dim_out: int):

        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.linear = nn.Linear(dim_in, dim_out)
        self.activation_function = nn.ReLU(inplace=True)
        self.batch_norm = nn.BatchNorm1d(dim_out)

    def forward(self, x):
        
        x = self.flatten(x)
        x = self.linear(x)
        x = self.activation_function(x)
        x = self.batch_norm(x)

        return x
    

class CrossSiameseNet(nn.Module):
    '''Siamese network using features from other siamese networks'''

    def __init__(self, models: List[nn.Module]):

        super().__init__()

        self.models = models
        self.n_models = len(models)
        self.cf_size = models[0].cf_size

        self.conv_block1 = ConvBlock(self.n_models, 64)
        self.conv_block2 = ConvBlock(64, 64)
        self.conv_block3 = ConvBlock(64, 64)
        self.conv_block4 = ConvBlock(64, 64)
        self.conv_block5 = ConvBlock(64, 64)
        self.conv_block6 = ConvBlock(64, 2)

        self.linear_block = LinearBlock(4*self.cf_size, 2*self.cf_size)

        # turn off grads in all parameters 
        for model in self.models:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

        # initialize the weights
        for conv_block in [self.conv_block1, self.conv_block2, self.conv_block3, 
                           self.conv_block4, self.conv_block5, self.conv_block6]:
            torch.nn.init.xavier_uniform_(conv_block.conv.weight)
            conv_block.conv.bias.data.fill_(0.01)

        torch.nn.init.xavier_uniform_(self.linear_block.linear.weight)
        self.linear_block.linear.bias.data.fill_(0.01)

    def forward_once(self, x):

        # features collected across all models
        features_submodels = [model.forward_once(x) for model in self.models]
        features_submodels = torch.stack(features_submodels, dim=-2)

        x = self.conv_block1(features_submodels)
        residual_features0 = x

        x = self.conv_block2(x)
        x = self.conv_block3(x, residual_features0)
        residual_features1 = x
        
        x = self.conv_block4(x)
        x = self.conv_block5(x, residual_features1)
        x = self.conv_block6(x)
        x = self.linear_block(x)

        return x

    def forward(self, x):
        return self.forward_once(x)