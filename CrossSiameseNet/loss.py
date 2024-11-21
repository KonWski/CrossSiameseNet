import torch.nn as nn
import torch
from sklearn.metrics import precision_score, accuracy_score, recall_score
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier

class WeightedTripletMarginLoss(nn.Module):
    
    def __init__(self, device, batch_size: int, weights_1: float, margin: float = 1, reduction_type: str = "mean"):
        super(WeightedTripletMarginLoss, self).__init__()
        self.device = device
        self.weights_1 = weights_1
        self.margin = margin
        self.reduction_type = reduction_type
        self.distance_metric = nn.PairwiseDistance()
        self.batch_size = batch_size
        self._tensor_zero = torch.zeros(self.batch_size).to(self.device)
    
    def forward(self, anchor_mf, positive_mf, negative_mf, anchor_label):

        dist_anch_pos = self.distance_metric(anchor_mf, positive_mf)
        dist_anch_neg = self.distance_metric(anchor_mf, negative_mf)
        weights = torch.where(anchor_label == 1, self.weights_1, 1)
        weights = weights.to(self.device)

        loss = torch.max(dist_anch_pos - dist_anch_neg + self.margin, self._tensor_zero[:dist_anch_neg.shape[0]]) * weights

        if self.reduction_type == "mean":
            loss = loss.mean()
        elif self.reduction_type == "sum":
            loss = loss.sum()
            
        return loss
    

# class WeightedTripletMarginLossNew(nn.Module):
    
#     def __init__(self, device, batch_size: int, weight_scenario: str, train_loader: DataLoader, margin: float = 1, reduction_type: str = "mean"):
#         super(WeightedTripletMarginLoss, self).__init__()
#         self.device = device
#         self.weight_scenario = weight_scenario
#         self.weights_1 = self._set_weights_1(train_loader)
#         self.margin = margin
#         self.reduction_type = reduction_type
#         self.distance_metric = nn.PairwiseDistance()
#         self.batch_size = batch_size
#         self._tensor_zero = torch.zeros(self.batch_size).to(self.device)

#     def _set_weights_1(self, train_loader):
        
#         if self.weight_scenario == "stable_weight_1":
#             weights_1 = 1.0
#         elif self.weight_scenario == "stable_boosted_weight_1":
#             weights_1 = len(train_loader.dataset.indices_0) / len(train_loader.dataset.indices_1)
#         else:
#             weights_1 = None

#         return weights_1

#     def forward(self, anchor_mf, positive_mf, negative_mf, anchor_label):
        
#         if self.weight_scenario in ["stable_weight_1", "stable_boosted_weight_1"]:
#             loss = self.forward_stable_weight_1(self, anchor_mf, positive_mf, negative_mf, anchor_label)

#         elif self.weight_scenario == "knn_weight_1":
        
#         return loss

#     def forward_stable_weight_1(self, anchor_mf, positive_mf, negative_mf, anchor_label):

#         dist_anch_pos = self.distance_metric(anchor_mf, positive_mf)
#         dist_anch_neg = self.distance_metric(anchor_mf, negative_mf)
#         weights = torch.where(anchor_label == 1, self.weights_1, 1)
#         weights = weights.to(self.device)

#         loss = torch.max(dist_anch_pos - dist_anch_neg + self.margin, self._tensor_zero[:dist_anch_neg.shape[0]]) * weights

#         if self.reduction_type == "mean":
#             loss = loss.mean()
#         elif self.reduction_type == "sum":
#             loss = loss.sum()
            
#         return loss

#     def refresh_knn_model(self, model, train_dataset, test_dataset):
        
#         model.eval()
#         y_train = train_dataset.y.numpy()
#         y_test = test_dataset.y.numpy()

#         train_embeddings = self._generate_embeddings(model, train_dataset, 1000)
#         test_embeddings = self._generate_embeddings(model, test_dataset, 1000)
#         knn = KNeighborsClassifier(n_neighbors=3)
#         knn.fit(train_embeddings.cpu().detach().numpy(), y_train)

#         # predictions
#         y_pred = knn.predict(train_embeddings.cpu().detach().numpy())

#         # scores
#         accuracy = round(accuracy_score(y_train, y_pred), 4)
#         precision = round(precision_score(y_train, y_pred), 4)
#         recall = round(recall_score(y_train, y_pred), 4)

#         print(f"TRAIN")
#         print(f"accuracy: {accuracy}, precision: {precision}, recall: {recall}")

#         # predictions
#         y_pred = knn.predict(test_embeddings.cpu().detach().numpy())

#         # scores
#         accuracy = round(accuracy_score(y_test, y_pred), 4)
#         precision = round(precision_score(y_test, y_pred), 4)
#         recall = round(recall_score(y_test, y_pred), 4)

#         print(f"TEST")
#         print(f"accuracy: {accuracy}, precision: {precision}, recall: {recall}")

#         # update
#         self.knn_model = knn

#     def _generate_embeddings(self, model, dataset, batch_size):
#         loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
#         embeddings = []
#         for batch_id, (anchor_mf, positive_mf, negative_mf, anchor_label) in enumerate(loader):
#             anchor_mf = anchor_mf.to(self.device)
#             embeddings.append(model(anchor_mf).cpu())
#         return torch.cat(embeddings, dim=0)