from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import torch
from torch.utils.data import DataLoader
from CrossSiameseNet.SiameseMolNet import SiameseMolNet, SiameseMolNetRegression, SiameseMolNetPretrained
from CrossSiameseNet.CrossSiameseNet import CrossSiameseNet
from skfp.metrics import enrichment_factor

def evaluate(model, train_dataset, test_dataset, y_train, y_test, device):

    train_embeddings = generate_embeddings(model, train_dataset, 1000, device)
    test_embeddings = generate_embeddings(model, test_dataset, 1000, device)

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
    roc_auc_train = round(roc_auc_score(y_train, y_pred), 4)
    mcc_train = round(matthews_corrcoef(y_train, y_pred), 4)

    ef01_train = round(enrichment_factor(y_train, y_pred, fraction=0.01), 4)
    ef05_train = round(enrichment_factor(y_train, y_pred, fraction=0.05), 4)
    ef10_train = round(enrichment_factor(y_train, y_pred, fraction=0.1), 4)
    ef15_train = round(enrichment_factor(y_train, y_pred, fraction=0.15), 4)
    ef20_train = round(enrichment_factor(y_train, y_pred, fraction=0.2), 4)

    print(f"TRAIN")
    print(f"accuracy: {accuracy_train}, precision: {precision_train}, recall: {recall_train}, f1_score: {f1_train}")
    print(f"ef01: {ef01_train}, ef05: {ef05_train}, ef10: {ef10_train}, ef15: {ef15_train}, ef20: {ef20_train}")
    print(f"roc_auc: {roc_auc_train}, mcc_train: {mcc_train}")

    # predictions
    y_pred = knn_test.predict(test_embeddings.cpu().detach().numpy())

    # scores
    accuracy_test = round(accuracy_score(y_test, y_pred), 4)
    precision_test = round(precision_score(y_test, y_pred), 4)
    recall_test = round(recall_score(y_test, y_pred), 4)
    f1_test = round(f1_score(y_test, y_pred), 4)
    roc_auc_test = round(roc_auc_score(y_test, y_pred), 4)
    mcc_test = round(matthews_corrcoef(y_test, y_pred), 4)

    ef01_test = round(enrichment_factor(y_test, y_pred, fraction=0.01), 4)
    ef05_test = round(enrichment_factor(y_test, y_pred, fraction=0.05), 4)
    ef10_test = round(enrichment_factor(y_test, y_pred, fraction=0.1), 4)
    ef15_test = round(enrichment_factor(y_test, y_pred, fraction=0.15), 4)
    ef20_test = round(enrichment_factor(y_test, y_pred, fraction=0.2), 4)

    print(f"TEST")
    print(f"accuracy: {accuracy_test}, precision: {precision_test}, recall: {recall_test}, f1_score: {f1_test}")
    print(f"ef01: {ef01_test}, ef05: {ef05_test}, ef10: {ef10_test}, ef15: {ef15_test}, ef20: {ef20_test}")
    print(f"roc_auc: {roc_auc_test}, mcc_test: {mcc_test}")
    print(8*"-")

    return accuracy_train, precision_train, recall_train, f1_train, ef01_train, roc_auc_train, roc_auc_train, \
        accuracy_test, precision_test, recall_test, f1_test, ef01_test, roc_auc_test, roc_auc_test


def generate_embeddings(model, dataset, batch_size, device):

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    embeddings = []

    for batch_id, (anchor_mf, _, _, _, _, _, _) in enumerate(loader):
        anchor_mf = anchor_mf.to(device)
        embedding_chunk = model(anchor_mf)

        anchor_mf = anchor_mf.detach().cpu()
        embedding_chunk = embedding_chunk.detach().cpu()

        embeddings.append(embedding_chunk)

    return torch.cat(embeddings, dim=0)


def load_dummy_model(model_name: str, csn_type: str = None, cf_size: int = 2048):

    if model_name == "SMN_HIV":
        dummy_model = SiameseMolNet(cf_size)
    elif model_name == "SMN_HIV_PRETRAINED":
        dummy_model = SiameseMolNetPretrained(cf_size, SiameseMolNetRegression(cf_size))
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

        if not csn_type:
            dummy_model = CrossSiameseNet(submodels)
        elif csn_type == "shorterVer0":
            dummy_model = CrossSiameseNetShorterVer0(submodels)
        elif csn_type == "shorterVer1":
            dummy_model = CrossSiameseNetShorterVer1(submodels)
        elif csn_type == "shorterVer2":
            dummy_model = CrossSiameseNetShorterVer2(submodels)
        elif csn_type == "shorterVer3":
            dummy_model = CrossSiameseNetShorterVer3(submodels)
        elif csn_type == "biggerVer0":
            dummy_model = CrossSiameseNetBiggerVer0(submodels)
        elif csn_type == "alternativeVer0":
            dummy_model = CrossSiameseNetAlternativeVer0(submodels)
        else:
            raise Exception(f"{csn_type} no implemented")

    return dummy_model


def send_model_to_device(model, device):

    if hasattr(model, "models"):
        for submodel in model.models:
            submodel.to(device)

    model.to(device)

    return model