from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, QED, Lipinski
import numpy as np
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen
import pandas as pd
from ogb.graphproppred.dataset_pyg import PygGraphPropPredDataset
import logging

def load_dataset(dataset_name, featurizer, splitter = None, ogbg_dataset_path = None):
    
    if dataset_name == "hiv":
        from deepchem.molnet import load_hiv
        _, datasets, _ = load_hiv(featurizer, splitter)

    if dataset_name == "ogbg-molhiv":
        datasets = load_ogbg_dataset(dataset_name, ogbg_dataset_path, featurizer)

    elif dataset_name == "hiv_esol":
        datasets = load_hiv_esol(featurizer, splitter)

    elif dataset_name == "hiv_tpsa":
        datasets = load_hiv_tpsa(featurizer, splitter)

    elif dataset_name == "hiv_logp":
        datasets = load_hiv_logp(featurizer, splitter)

    elif dataset_name == "hiv_molmr":
        datasets = load_hiv_molmr(featurizer, splitter)

    elif dataset_name == "hiv_molwt":
        datasets = load_hiv_extra_param(ogbg_dataset_path, featurizer, "molwt")

    elif dataset_name == "hiv_num_rotatable_bonds":
        datasets = load_hiv_extra_param(ogbg_dataset_path, featurizer, "num_rotatable_bonds")

    elif dataset_name == "hiv_num_rings":
        datasets = load_hiv_extra_param(ogbg_dataset_path, featurizer, "num_rings")

    elif dataset_name == "hiv_num_h_bonds_acceptors":
        datasets = load_hiv_extra_param(ogbg_dataset_path, featurizer, "num_h_bonds_acceptors")

    elif dataset_name == "hiv_heavy_atom_count":
        datasets = load_hiv_extra_param(ogbg_dataset_path, featurizer, "heavy_atom_count")

    elif dataset_name == "hiv_csp3":
        datasets = load_hiv_extra_param(ogbg_dataset_path, featurizer, "csp3")

    elif dataset_name == "hiv_asa":
        datasets = load_hiv_extra_param(ogbg_dataset_path, featurizer, "asa")

    elif dataset_name == "hiv_num_valence_electrons":
        datasets = load_hiv_extra_param(ogbg_dataset_path, featurizer, "num_valence_electrons")

    elif dataset_name == "hiv_num_radical_electrons":
        datasets = load_hiv_extra_param(ogbg_dataset_path, featurizer, "num_radical_electrons")

    elif dataset_name == "hiv_formal_charge":
        datasets = load_hiv_extra_param(ogbg_dataset_path, featurizer, "formal_charge")

    elif dataset_name == "hiv_qed":
        datasets = load_hiv_extra_param(ogbg_dataset_path, featurizer, "qed")

    elif dataset_name == "delaney":
        from deepchem.molnet import load_delaney
        _, datasets, _ = load_delaney(featurizer, splitter)
    
    elif dataset_name == "lipo":
        from deepchem.molnet import load_lipo
        _, datasets, _ = load_lipo(featurizer, splitter)
    
    elif dataset_name[:5] == "tox21":
        from deepchem.molnet import load_tox21
        task = dataset_name[dataset_name.find("_")+1:]
        _, datasets, _ = load_tox21(featurizer, splitter, tasks=[task])
    
    else:
        raise Exception(f"Dataset {dataset_name} not implemented")

    train_dataset, val_dataset, test_dataset = datasets
    X_train, y_train, smiles_train = train_dataset.X, train_dataset.y, train_dataset.ids
    X_val, y_val, smiles_val = val_dataset.X, val_dataset.y, val_dataset.ids
    X_test, y_test, smiles_test = test_dataset.X, test_dataset.y, test_dataset.ids

    return X_train, y_train, smiles_train, X_val, y_val, smiles_val, X_test, y_test, smiles_test


def load_hiv_extra_param(dataset_path, featurizer, param_name):

    _, datasets, _ = load_ogbg_dataset("ogbg-molhiv", dataset_path, featurizer)
    train_dataset, val_dataset, test_dataset = datasets

    X_train, smiles_train = train_dataset.X, train_dataset.ids
    X_val, smiles_val = val_dataset.X, val_dataset.ids
    X_test, smiles_test = test_dataset.X, test_dataset.ids

    y_train = np.array([calculate_extra_param(smiles, param_name) for smiles in smiles_train])
    y_val = np.array([calculate_extra_param(smiles, param_name) for smiles in smiles_val])
    y_test = np.array([calculate_extra_param(smiles, param_name) for smiles in smiles_test])
    
    return DummyDataset(X_train, y_train, smiles_train), \
        DummyDataset(X_val, y_val, smiles_val), \
        DummyDataset(X_test, y_test, smiles_test)


def load_ogbg_dataset(dataset, dataset_path, featurizer):

    ogbg_dataset = PygGraphPropPredDataset(name = dataset, root = dataset_path)

    train_ids = ogbg_dataset.get_idx_split()["train"]
    train_dataset = ogbg_dataset[train_ids]
    smiles_train = get_smiles(train_dataset.root, train_ids)
    X_train, y_train, smiles_train = load_ecfp_fingerprints(smiles_train, train_dataset.y, featurizer)

    valid_ids = ogbg_dataset.get_idx_split()["valid"]
    valid_dataset = ogbg_dataset[valid_ids]
    smiles_valid = get_smiles(valid_dataset.root, valid_ids)
    X_valid, y_valid, smiles_valid = load_ecfp_fingerprints(smiles_valid, valid_dataset.y, featurizer)

    test_ids = ogbg_dataset.get_idx_split()["test"]
    test_dataset = ogbg_dataset[test_ids]
    smiles_test = get_smiles(test_dataset.root, test_ids)
    X_test, y_test, smiles_test = load_ecfp_fingerprints(smiles_test, test_dataset.y, featurizer)

    return DummyDataset(X_train, y_train, smiles_train), \
        DummyDataset(X_valid, y_valid, smiles_valid), \
        DummyDataset(X_test, y_test, smiles_test)


def get_smiles(mapping_path, molecule_ids):
    df = pd.read_csv(f"{mapping_path}/mapping/mol.csv.gz")
    return df.loc[molecule_ids]["smiles"].to_list()


def load_ecfp_fingerprints(smiles, y, featurizer):

    X = []
    updated_y = []
    updated_smiles = []

    for smile, label in zip(smiles, y):
        
        mol = Chem.MolFromSmiles(smile)

        if mol is not None:
            mol = Chem.AddHs(mol)
            fingerprint = featurizer.GetFingerprintAsNumPy(mol)

            X.append(fingerprint)
            updated_y.append(label)
            updated_smiles.append(smile)

        else:
            logging.info(f"Rdkit was not able to convert Smile {smile} to a mol. Hash used as a scaffold.")

    return np.array(X), updated_y, updated_smiles


def calculate_extra_param(smiles, param_name):

    mol = Chem.MolFromSmiles(smiles)
    if param_name == "molwt":
        param_value = Descriptors.MolWt(mol)
    elif param_name == "num_rotatable_bonds":
        param_value = Lipinski.NumRotatableBonds(mol)
    elif param_name == "num_rings":
        param_value = rdMolDescriptors.CalcNumRings(mol)
    elif param_name == "num_h_bonds_acceptors":
        param_value = Lipinski.NumHAcceptors(mol)
    elif param_name == "heavy_atom_count":
        param_value = rdMolDescriptors.CalcNumHeavyAtoms(mol)
    elif param_name == "csp3":
        param_value = rdMolDescriptors.CalcFractionCSP3(mol)
    elif param_name == "asa":
        param_value = rdMolDescriptors.CalcLabuteASA(mol)
    elif param_name == "num_valence_electrons":
        param_value = Descriptors.NumValenceElectrons(mol)
    elif param_name == "num_radical_electrons":
        param_value = Descriptors.NumRadicalElectrons(mol)
    elif param_name == "formal_charge":
        param_value = Chem.GetFormalCharge(mol)
    elif param_name == "qed":
        param_value = QED.qed(mol)
    else:
        raise Exception(f"Not implemented param: {param_name}")

    return param_value


def load_hiv_esol(featurizer, splitter):
    _, datasets, _ = load_ogbg_dataset(featurizer, splitter)
    train_dataset, val_dataset, test_dataset = datasets

    X_train, smiles_train = train_dataset.X, train_dataset.ids
    X_val, smiles_val = val_dataset.X, val_dataset.ids
    X_test, smiles_test = test_dataset.X, test_dataset.ids

    y_train = np.array([esol_predict(smiles) for smiles in smiles_train])
    y_val = np.array([esol_predict(smiles) for smiles in smiles_val])
    y_test = np.array([esol_predict(smiles) for smiles in smiles_test])
    
    return DummyDataset(X_train, y_train, smiles_train), \
        DummyDataset(X_val, y_val, smiles_val), \
        DummyDataset(X_test, y_test, smiles_test)
            

def esol_predict(smiles: str):

    mol = Chem.MolFromSmiles(smiles)

    # octanol-water partition coefficient, molecular weight, rotatable bonds 
    logP = Crippen.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    aromatic_proportion = (
        sum(1 for a in mol.GetAromaticAtoms()) / mol.GetNumAtoms()
        if mol.GetNumAtoms() > 0 else 0
    )

    esol_solubility = 0.16 - 0.63 * logP - 0.0062 * mw + 0.066 * rot_bonds - 0.74 * aromatic_proportion

    return esol_solubility


def load_hiv_tpsa(featurizer, splitter):
    '''topological polar surface'''

    _, datasets, _ = load_ogbg_dataset(featurizer, splitter)
    train_dataset, val_dataset, test_dataset = datasets

    X_train, smiles_train = train_dataset.X, train_dataset.ids
    X_val, smiles_val = val_dataset.X, val_dataset.ids
    X_test, smiles_test = test_dataset.X, test_dataset.ids

    y_train = np.array([compute_tpsa(smiles) for smiles in smiles_train])
    y_val = np.array([compute_tpsa(smiles) for smiles in smiles_val])
    y_test = np.array([compute_tpsa(smiles) for smiles in smiles_test])
    
    return DummyDataset(X_train, y_train, smiles_train), \
        DummyDataset(X_val, y_val, smiles_val), \
        DummyDataset(X_test, y_test, smiles_test)


def compute_tpsa(smiles: str):

    mol = Chem.MolFromSmiles(smiles)
    tpsa = Descriptors.TPSA(mol)
    return tpsa


def load_hiv_logp(featurizer, splitter):
    '''Octanol-water partition coefficient (lipophilicity)'''

    _, datasets, _ = load_ogbg_dataset(featurizer, splitter)
    train_dataset, val_dataset, test_dataset = datasets

    X_train, smiles_train = train_dataset.X, train_dataset.ids
    X_val, smiles_val = val_dataset.X, val_dataset.ids
    X_test, smiles_test = test_dataset.X, test_dataset.ids

    y_train = np.array([compute_logp(smiles) for smiles in smiles_train])
    y_val = np.array([compute_logp(smiles) for smiles in smiles_val])
    y_test = np.array([compute_logp(smiles) for smiles in smiles_test])
    
    return DummyDataset(X_train, y_train, smiles_train), \
        DummyDataset(X_val, y_val, smiles_val), \
        DummyDataset(X_test, y_test, smiles_test)


def compute_logp(smiles: str):

    mol = Chem.MolFromSmiles(smiles)
    logp = Descriptors.MolLogP(mol)
    return logp


def load_hiv_molmr(featurizer, splitter):
    '''Crippen’s Molar Refractivity (MR)'''

    _, datasets, _ = load_ogbg_dataset(featurizer, splitter)
    train_dataset, val_dataset, test_dataset = datasets

    X_train, smiles_train = train_dataset.X, train_dataset.ids
    X_val, smiles_val = val_dataset.X, val_dataset.ids
    X_test, smiles_test = test_dataset.X, test_dataset.ids

    y_train = np.array([compute_molmr(smiles) for smiles in smiles_train])
    y_val = np.array([compute_molmr(smiles) for smiles in smiles_val])
    y_test = np.array([compute_molmr(smiles) for smiles in smiles_test])
    
    return DummyDataset(X_train, y_train, smiles_train), \
        DummyDataset(X_val, y_val, smiles_val), \
        DummyDataset(X_test, y_test, smiles_test)


def compute_molmr(smiles: str):

    mol = Chem.MolFromSmiles(smiles)
    molmr = Crippen.MolMR(mol)
    return molmr


class DummyDataset():

    def __init__(self, X, y, smiles):
        self.X = X
        self.y = y
        self.ids = smiles