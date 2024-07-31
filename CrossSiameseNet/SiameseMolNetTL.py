import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseMolNetTriplet(nn.Module):
    """
    Siamese Network designed to be used together with triplet loss
    """
    def __init__(self, cf_size: int):

        super().__init__()

        self.cf_size = cf_size
        self.linear_1 = nn.Linear(cf_size, 2*cf_size)
        self.batch_norm_1 = nn.BatchNorm1d(2*cf_size)

        self.linear_2 = nn.Linear(2*cf_size, 2*cf_size)
        self.batch_norm_2 = nn.BatchNorm1d(2*cf_size)

        self.linear_3 = nn.Linear(2*cf_size, 2*cf_size)
        self.batch_norm_3 = nn.BatchNorm1d(2*cf_size)

        # initialize the weights
        for layer in [self.linear_1, self.linear_2, self.linear_3, 
                      self.linear_output_1, self.linear_output_2, self.linear_output_3]:
            
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)

    def forward(self, x):

        features = F.relu(self.linear_1(x))
        features = self.batch_norm_1(features)

        features = F.relu(self.linear_2(features))
        features = self.batch_norm_2(features)

        features = F.relu(self.linear_3(features))
        features = self.batch_norm_3(features)

        return features