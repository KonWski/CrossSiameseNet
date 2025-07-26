import torch
from CrossSiameseNet.datasets.cf_datasets import get_dataset, MolDataset, MolDatasetTriplet
from CrossSiameseNet.CrossSiameseNet import CrossSiameseNet
from CrossSiameseNet.SiameseMolNet import SiameseMolNet, SiameseMolNetRegression
import deepchem as dc
from CrossSiameseNet.checkpoints import load_checkpoint
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
import gc
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
import pandas as pd

def generate_embeddings(model, dataset, batch_size, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    embeddings = []
    for batch_id, (anchor_mf, positive_mf, negative_mf, anchor_label) in enumerate(loader):
        anchor_mf = anchor_mf.to(device)
        embeddings.append(model(anchor_mf).cpu())
    return torch.cat(embeddings, dim=0)


def test_net(model, train_dataset, test_dataset, val_dataset, device):

    train_accuracy = []
    train_precision = []
    train_recall = []
    test_accuracy = []
    test_precision = []
    test_recall = []

    train_embeddings, labels = generate_embeddings(model, train_dataset, 1000)
    test_embeddings, labels = generate_embeddings(model, test_dataset, 1000)
    val_embeddings, labels = generate_embeddings(model, test_dataset, 1000)

    y_train = train_dataset.y
    y_test = test_dataset.y
    y_val = val_dataset.y

    # fit model
    knn = KNeighborsClassifier(n_neighbors=4)
    knn.fit(train_embeddings.cpu().detach().numpy(), y_train)

    knn_test = KNeighborsClassifier(n_neighbors=3)
    knn_test.fit(train_embeddings.cpu().detach().numpy(), y_train)

    # predictions
    y_pred_proba = knn.predict_proba(train_embeddings.cpu().detach().numpy())
    y_proba_update = np.zeros([y_train.shape[0], 2])

    for row_id in range(y_train.shape[0]):
        true_label = int(y_train[row_id][0])
        y_proba_update[row_id, true_label] = 0.25

    y_pred_proba = y_pred_proba - y_proba_update
    y_pred = np.zeros(y_pred_proba.shape[0])
    full_proba = y_pred_proba[:,1] >= 0.5
    y_pred[full_proba] = 1

    # scores
    train_accuracy = round(accuracy_score(y_train, y_pred), 4)
    train_precision = round(precision_score(y_train, y_pred), 4)
    train_recall = round(recall_score(y_train, y_pred), 4)
    train_f_1 = round(f1_score(y_train, y_pred), 4)

    print(f"TRAIN")
    print(f"accuracy: {train_accuracy}, precision: {train_precision}, recall: {train_recall}, F1: {train_f_1}")

    # predictions
    y_pred = knn_test.predict(test_embeddings.cpu().detach().numpy())

    # scores
    test_accuracy = round(accuracy_score(y_test, y_pred), 4)
    test_precision = round(precision_score(y_test, y_pred), 4)
    test_recall = round(recall_score(y_test, y_pred), 4)
    test_f_1 = round(f1_score(y_test, y_pred), 4)

    print(f"TEST")
    print(f"accuracy: {test_accuracy}, precision: {test_precision}, recall: {test_recall}, F1: {test_f_1}")
    print(8*"-")

    # predictions
    y_pred = knn_test.predict(val_embeddings.cpu().detach().numpy())

    # scores
    val_accuracy = round(accuracy_score(y_val, y_pred), 4)
    val_precision = round(precision_score(y_val, y_pred), 4)
    val_recall = round(recall_score(y_val, y_pred), 4)
    val_f_1 = round(f1_score(y_val, y_pred), 4)

    print(f"VALIDATION")
    print(f"accuracy: {val_accuracy}, precision: {val_precision}, recall: {val_recall}, F1: {val_f_1}")
    print(8*"-")

    return train_accuracy, train_precision, train_recall, train_f_1, \
        test_accuracy, test_precision, test_recall, test_f_1, \
            val_accuracy, val_precision, val_recall, val_f_1