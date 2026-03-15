import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.neighbors import KNeighborsClassifier
from skfp.metrics import enrichment_factor
import logging
from CrossSiameseNet.CrossSiameseNet import CrossSiameseNet
import pandas as pd

class Statistics:

    def __init__(self, device, experiment_hash, model_name, n_epochs):
        self.device = device
        self.experiment_hash = experiment_hash
        self.model_name = model_name
        self.n_epochs = n_epochs
        self.accumulated_statistics = {
            "epoch_id": [{
                "train": {
                    "loss": None,
                    "distances_0_0_mean": None,
                    "distances_1_1_mean": None,
                    "distances_0_1_mean": None,
                    "accuracy": None,
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "ef01": None,
                    "ef05": None,
                    "roc_auc": None,
                    "mcc": None
                },
                "test": {
                    "loss": None,
                    "distances_0_0_mean": None,
                    "distances_1_1_mean": None,
                    "distances_0_1_mean": None,
                    "accuracy": None,
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "ef01": None,
                    "ef05": None,
                    "roc_auc": None,
                    "mcc": None
                }
            } for _ in range(n_epochs)]
        }

    def refresh_embeddings(self, model, train_loader, test_loader):

        train_anchors_transformed, train_anchor_labels = self._generate_embeddings(model, train_loader.dataset, 100)
        self.train_embeddings = train_anchors_transformed.detach()
        self.train_labels = train_anchor_labels.detach()

        test_anchors_transformed, test_anchor_labels = self._generate_embeddings(model, test_loader.dataset, 100)
        self.test_embeddings = test_anchors_transformed.detach()
        self.test_labels = test_anchor_labels.detach()


    def _generate_embeddings(self, model, dataset, batch_size):

        model.eval()
        if isinstance(model, CrossSiameseNet):
            for m in model.models:
                m.eval()

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        embeddings = []
        anchor_labels = []

        with torch.no_grad():
            for batch_id, (anchor_mf, _, _, anchor_label, _, _, _) in enumerate(loader):
                anchor_mf = anchor_mf.to(self.device)
                embeddings.append(model(anchor_mf).cpu())
                anchor_labels.append(anchor_label.cpu())

        return torch.cat(embeddings, dim=0), torch.cat(anchor_labels, dim=0)
    

    def update_statistics(self, epoch_id, train_loss):

        for state in ["train", "test"]:
            distances_0_0_mean, distances_1_1_mean, distances_0_1_mean = self._distance_stats(state)
            self.accumulated_statistics["epoch_id"][epoch_id][state]["distances_0_0_mean"] = distances_0_0_mean
            self.accumulated_statistics["epoch_id"][epoch_id][state]["distances_1_1_mean"] = distances_1_1_mean
            self.accumulated_statistics["epoch_id"][epoch_id][state]["distances_0_1_mean"] = distances_0_1_mean

        accuracy, precision, recall, f1, ef01, ef05, roc_auc, mcc = self._evaluate_model(n_neighbors=10)
        self.accumulated_statistics["epoch_id"][epoch_id]["test"]["accuracy"] = accuracy
        self.accumulated_statistics["epoch_id"][epoch_id]["test"]["precision"] = precision
        self.accumulated_statistics["epoch_id"][epoch_id]["test"]["recall"] = recall
        self.accumulated_statistics["epoch_id"][epoch_id]["test"]["f1"] = f1
        self.accumulated_statistics["epoch_id"][epoch_id]["test"]["ef01"] = ef01
        self.accumulated_statistics["epoch_id"][epoch_id]["test"]["ef05"] = ef05
        self.accumulated_statistics["epoch_id"][epoch_id]["test"]["roc_auc"] = roc_auc
        self.accumulated_statistics["epoch_id"][epoch_id]["test"]["mcc"] = mcc

        # manual stats update
        self.accumulated_statistics["epoch_id"][epoch_id]["train"]["loss"] = train_loss
        self.accumulated_statistics["epoch_id"][epoch_id]["test"]["loss"] = None


    def _distance_stats(self, state):

        if state == "train":
            anchors_transformed = self.train_embeddings
            anchor_labels = self.train_labels
        else:
            anchors_transformed = self.test_embeddings
            anchor_labels = self.test_labels

        indices_1 = (anchor_labels == 1).nonzero()[:,0].tolist()
        indices_0 = (anchor_labels == 0).nonzero()[:,0].tolist()

        distances = torch.cdist(anchors_transformed, anchors_transformed)

        distances_0_0 = distances[indices_0, indices_0]
        distances_0_0 = distances_0_0[distances_0_0 != 0]
        distances_0_0_mean = round(torch.mean(distances_0_0).item(), 5)

        distances_1_1 = distances[indices_1, indices_1]
        distances_1_1 = distances_1_1[distances_1_1 != 0]
        distances_1_1_mean = round(torch.mean(distances_1_1).item(), 5)

        distances_0_1 = []
        for i0 in indices_0:
            for i1 in indices_1:
                distances_0_1.append(distances[i0, i1].item())
        distances_0_1_mean = round(sum(distances_0_1) / len(distances_0_1), 5)

        return distances_0_0_mean, distances_1_1_mean, distances_0_1_mean
    

    def _evaluate_model(self, n_neighbors):

        # fit model
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(self.train_embeddings, self.train_labels)

        # predictions
        y_pred = knn.predict(self.test_embeddings)

        # scores
        accuracy = round(accuracy_score(self.test_labels, y_pred), 4)
        precision = round(precision_score(self.test_labels, y_pred), 4)
        recall = round(recall_score(self.test_labels, y_pred), 4)
        f1 = round(f1_score(self.test_labels, y_pred), 4)
        roc_auc = round(roc_auc_score(self.test_labels, y_pred), 4)
        mcc = round(matthews_corrcoef(self.test_labels, y_pred), 4)

        ef01 = round(enrichment_factor(self.test_labels, y_pred, fraction=0.01), 4)
        ef05 = round(enrichment_factor(self.test_labels, y_pred, fraction=0.05), 4)

        return accuracy, precision, recall, f1, ef01, ef05, roc_auc, mcc
    

    def log_statistics(self, epoch_id):
        
        epoch_statistics = self.accumulated_statistics["epoch_id"][epoch_id]
        logging.info(f"Epoch_id: {epoch_id}, State: train, Stats: {epoch_statistics['train']}")
        logging.info(f"Epoch_id: {epoch_id}, State: test, Stats: {epoch_statistics['test']}")


    def save_statistics(self, path):
        rows = []

        for epoch_id in range(self.n_epochs):
            
            epoch_stats = self.accumulated_statistics["epoch_id"][epoch_id]
            row = {"epoch": epoch_id}

            for state in ["train", "test"]:
                for metric, value in epoch_stats[state].items():
                    row[f"{state}_{metric}"] = value

            rows.append(row)

        df = pd.DataFrame(rows)
        df["experiment_hash"] = self.experiment_hash
        df["model_name"] = self.model_name

        df.to_excel(path, index=False)

    
    def get_metric_value(self, metric_name, state, epoch_id):
        return self.accumulated_statistics["epoch_id"][epoch_id][state][metric_name]