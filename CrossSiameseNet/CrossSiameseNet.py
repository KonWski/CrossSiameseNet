import torch
import torch.nn as nn
from typing import List


class ConvBlock(nn.Module):

    def __init__(self, dim_in: int, dim_out: int):

        super().__init__()
        self.conv = nn.Conv1d(dim_in, dim_out, 1)
        self.activation_function = nn.ReLU(inplace=True)
        self.batch_norm = nn.BatchNorm1d(dim_out)
        self.dropout = nn.Dropout(p=0.2, inplace=True)

    def forward(self, x, residual = None):

        if residual is not None:
            x += residual

        x = self.conv(x)
        x = self.activation_function(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        return x


class LinearBlock(nn.Module):

    def __init__(self, dim_in: int, dim_out: int):

        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.linear = nn.Linear(dim_in, dim_out)
        self.activation_function = nn.ReLU(inplace=True)
        self.batch_norm = nn.BatchNorm1d(dim_out)
        self.dropout = nn.Dropout(p=0.2, inplace=True)

    def forward(self, x):
        
        x = self.flatten(x)
        print(f"linear0 x.shape {x.shape}")
        x = self.linear(x)
        print(f"linear1 x.shape {x.shape}")

        x = self.activation_function(x)
        print(f"linear2 x.shape {x.shape}")

        x = self.batch_norm(x)
        print(f"linear3 x.shape {x.shape}")

        x = self.dropout(x)
        print(f"linear4 x.shape {x.shape}")

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

    def forward_once(self, x):

        # features collected across all models
        features_submodels = [model.forward_once(x) for model in self.models]
        print(f"features_submodels[0].shape: {features_submodels[0].shape}")
        features_submodels = torch.stack(features_submodels, dim=-2)
        print(f"features_submodels.shape: {features_submodels.shape}")

        x = self.conv_block1(features_submodels)
        print(f"x0.shape: {x.shape}")

        residual_features0 = x
        x = self.conv_block2(x)
        print(f"x1.shape: {x.shape}")

        x = self.conv_block3(x, residual_features0)
        print(f"x2.shape: {x.shape}")

        residual_features1 = x
        x = self.conv_block4(x)
        print(f"x3.shape: {x.shape}")

        x = self.conv_block5(x, residual_features1)
        print(f"x4.shape: {x.shape}")

        x = self.conv_block6(x)
        print(f"x5.shape: {x.shape}")

        x = self.linear_block(x)
        print(f"x6.shape: {x.shape}")

        return x

    def forward(self, x):
        return self.forward_once(x)