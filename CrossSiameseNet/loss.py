import torch.nn as nn
import torch

class WeightedTripletMarginLoss(nn.Module):
    
    def __init__(self, weights_1: float, margin: float = 1, reduction_type: str = "mean"):
        super(WeightedTripletMarginLoss, self).__init__()
        self.weights_1 = weights_1
        self.margin = margin
        self.reduction_type = reduction_type
        self.distance_metric = nn.PairwiseDistance()
    
    def forward(self, anchor_mf, positive_mf, negative_mf, anchor_label):

        dist_anch_pos = self.distance_metric(anchor_mf, positive_mf)
        dist_anch_neg = self.distance_metric(anchor_mf, negative_mf)
        weights = torch.tensor([self.weights_1 for i in range(len(anchor_mf))]) * anchor_label

        loss = torch.max(dist_anch_pos - dist_anch_neg + self.margin, torch.zeros(len(anchor_mf)))
        loss = loss * weights

        if self.reduction_type == "mean":
            loss = loss.mean()
        elif self.reduction_type == "sum":
            loss = loss.sum()
            
        return loss