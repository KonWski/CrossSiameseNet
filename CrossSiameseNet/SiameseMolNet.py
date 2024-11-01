import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseMolNet(nn.Module):

    def __init__(self, cf_size: int):

        super().__init__()

        self.cf_size = cf_size
        self.linear_1 = nn.Linear(cf_size, 2*cf_size)
        self.batch_norm_1 = nn.BatchNorm1d(2*cf_size)

        self.linear_2 = nn.Linear(2*cf_size, 2*cf_size)
        self.batch_norm_2 = nn.BatchNorm1d(2*cf_size)

        self.linear_3 = nn.Linear(2*cf_size, 2*cf_size)
        self.batch_norm_3 = nn.BatchNorm1d(2*cf_size)


    def forward_once(self, x):

        features = F.relu(self.linear_1(x))
        features = self.batch_norm_1(features)

        features = F.relu(self.linear_2(features))
        features = self.batch_norm_2(features)

        features = F.relu(self.linear_3(features))
        features = self.batch_norm_3(features)

        return features

    def forward(self, x):
        return self.forward_once(x)
    

class SiameseMolNetRegression(SiameseMolNet):

    def __init__(self, cf_size: int):

        super().__init__(cf_size)

        self.linear_output_1 = nn.Linear(2*self.cf_size, self.cf_size)
        self.batch_norm_6 = nn.BatchNorm1d(self.cf_size)
        self.linear_output_2 = nn.Linear(self.cf_size, 64)
        self.batch_norm_7 = nn.BatchNorm1d(64)
        self.linear_output_3 = nn.Linear(64, 1)

        # initialize the weights
        for layer in [self.linear_1, self.linear_2, self.linear_3, 
                      self.linear_output_1, self.linear_output_2, self.linear_output_3]:
            
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)

    
    def forward(self, mol0, mol1):

        # process two molecules
        features0 = self.forward_once(mol0)
        features1 = self.forward_once(mol1)

        # combine both feature vectors
        features = torch.stack((features0, features1), 0)
        features_mean = torch.mean(features, 0)

        # final output
        output = F.relu(self.linear_output_1(features_mean))
        output = self.batch_norm_6(output)

        output = F.relu(self.linear_output_2(output))
        output = self.batch_norm_7(output)

        output = self.linear_output_3(output)

        return output