import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseMolNet(nn.Module):

    def __init__(self, cf_size: int, task: str):

        super().__init__()

        self.cf_size = cf_size
        self.task = task
        self.linear_1 = nn.Linear(cf_size, 2*cf_size)
        self.batch_norm_1 = nn.BatchNorm1d(2*cf_size)

        self.linear_2 = nn.Linear(2*cf_size, 2*cf_size)
        self.batch_norm_2 = nn.BatchNorm1d(2*cf_size)

        self.linear_3 = nn.Linear(2*cf_size, 2*cf_size)
        self.batch_norm_3 = nn.BatchNorm1d(2*cf_size)

        if self.task == "classification":
                self.linear_output = nn.Linear(64, 2)
        elif self.task == "regression":
                self.linear_output = nn.Linear(64, 1)

        # initialize the weights
        for layer in [self.linear_1, self.linear_2, self.linear_3, self.linear_output]:
            
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)


    def forward_once(self, x):

        features = F.relu(self.linear_1(x))
        features = self.batch_norm_1(features)

        features = F.relu(self.linear_2(features))
        features = self.batch_norm_2(features)

        features = F.relu(self.linear_3(features))
        features = self.batch_norm_3(features)

        return features


    def forward(self, x):

        features = self.forward_once(x)
        output = self.linear_output(features)

        return output