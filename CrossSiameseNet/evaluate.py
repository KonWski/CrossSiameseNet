from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import torch
from torch.utils.data import DataLoader
from CrossSiameseNet.SiameseMolNet import SiameseMolNet, SiameseMolNetRegression
from CrossSiameseNet.CrossSiameseNet import CrossSiameseNet

def evaluate(model, train_dataset, test_dataset, y_train, y_test, device):

    train_embeddings, _ = generate_embeddings(model, train_dataset, 1000, device)
    test_embeddings, _ = generate_embeddings(model, test_dataset, 1000, device)

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
    accuracy_train = round(accuracy_score(y_train, y_pred), 4)
    precision_train = round(precision_score(y_train, y_pred), 4)
    recall_train = round(recall_score(y_train, y_pred), 4)
    f1_train = round(f1_score(y_train, y_pred), 4)

    print(f"TRAIN")
    print(f"accuracy: {accuracy_train}, precision: {precision_train}, recall: {recall_train}, f1_score: {f1_train}")

    # predictions
    y_pred = knn_test.predict(test_embeddings.cpu().detach().numpy())

    # scores
    accuracy_test = round(accuracy_score(y_test, y_pred), 4)
    precision_test = round(precision_score(y_test, y_pred), 4)
    recall_test = round(recall_score(y_test, y_pred), 4)
    f1_test = round(f1_score(y_test, y_pred), 4)

    print(f"TEST")
    print(f"accuracy: {accuracy_test}, precision: {precision_test}, recall: {recall_test}, f1_score: {f1_test}")
    print(8*"-")

    return accuracy_train, precision_train, recall_train, f1_train, accuracy_test, precision_test, recall_test, f1_test


def generate_embeddings(model, dataset, batch_size, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    embeddings = []
    anchor_labels = []
    for batch_id, (anchor_mf, positive_mf, negative_mf, anchor_label) in enumerate(loader):
        anchor_mf = anchor_mf.to(device)
        model_anchor_mf = model(anchor_mf)
        embeddings.append(model_anchor_mf.cpu())
        anchor_labels.append(anchor_label.cpu())
    return torch.cat(embeddings, dim=0), torch.cat(embeddings, dim=0)


def load_dummy_model(model_name: str, cf_size: int = 2048):

    if model_name == "SMN_HIV":
        dummy_model = SiameseMolNet(cf_size)
    else:
        # first submodel
        smn_hiv = SiameseMolNet(cf_size)

        # second submodel
        submodel_name = model_name[12:]
        if len(submodel_name) > 0:

            if submodel_name in ["SMN_LIPO", "SMN_DELANEY", "SMN_HIV_ESOL"]:
                second_submodel = SiameseMolNetRegression(cf_size)
            else:
                second_submodel = SiameseMolNet(cf_size)

            submodels = [smn_hiv, second_submodel]
        else:
            submodels = [smn_hiv]

        dummy_model = CrossSiameseNet(submodels)

    return dummy_model