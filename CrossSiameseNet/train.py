import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import logging
from datetime import datetime
import pandas as pd
from CrossSiameseNet.checkpoints import save_checkpoint
from CrossSiameseNet.BatchShaper import BatchShaper
from CrossSiameseNet.loss import WeightedTripletMarginLoss
from CrossSiameseNet.Statistics import Statistics
from CrossSiameseNet.MoleculeAugmentator import MoleculeAugmentator
from CrossSiameseNet.CrossSiameseNet import CrossSiameseNet
from uuid import uuid4
import sys


def train_triplet(model, dataset_name: str, train_loader: DataLoader, test_loader: DataLoader, 
                  n_epochs: int, device, checkpoints_dir: str, use_fixed_training_triplets: bool = False,
                  training_type: str = None, alpha: float = None, weight_ones = True, 
                  molecule_augmentator: MoleculeAugmentator = None, lr: float = None):
    
    model = model.to(device)
    experiment_hash = uuid4().hex
    optimizer = Adam(model.parameters(), lr=lr)
    if weight_ones:
        weights_1 = len(train_loader.dataset.indices_0) / len(train_loader.dataset.indices_1)
    else:
        weights_1 = 1.0

    criterion_triplet_loss = WeightedTripletMarginLoss(device, train_loader.batch_size, weights_1)
    batch_shaper = BatchShaper(device, training_type, alpha)
    statistics = Statistics(device, experiment_hash, model.model_name, n_epochs)
    best_roc_auc_score = 0

    for epoch in range(0, n_epochs):
        
        losses = {"train": None, "test": None}

        # set fixed training dataset for models comparison
        if epoch > 0 and use_fixed_training_triplets:
                train_loader.dataset.refresh_fixed_triplets(train_loader.dataset.seed_fixed_triplets + epoch)

        for state, loader in zip(["train", "test"], [train_loader, test_loader]):
            
            # calculated parameters
            running_loss = 0.0

            if state == "train":
                model.train()

                if isinstance(model, CrossSiameseNet):
                    for m in model.models:
                        m.train()

                loader.dataset.shuffle_data(train_loader.batch_size)

                for batch_id, (anchor_mf, positive_mf, negative_mf, anchor_label, anchor_smiles, _, _) in enumerate(loader):

                    with torch.set_grad_enabled(state == 'train'):
                        
                        optimizer.zero_grad()

                        if molecule_augmentator and state == "train":
                            anchor_mf = molecule_augmentator.transform_batch(anchor_mf, anchor_smiles)
                            anchor_mf = anchor_mf.to(anchor_mf)

                        anchor_mf, positive_mf, negative_mf, anchor_label = batch_shaper.shape_batch(anchor_mf, positive_mf, negative_mf, anchor_label, model, state)
                        loss = criterion_triplet_loss(anchor_mf, positive_mf, negative_mf, anchor_label)

                        if state == "train":
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item()

                epoch_loss = round(running_loss / (batch_id + 1), 5)
                losses[state] = epoch_loss

            else:
                model.eval()
                if isinstance(model, CrossSiameseNet):
                    for m in model.models:
                        m.eval()

        statistics.refresh_embeddings(model, train_loader, test_loader)
        statistics.update_statistics(epoch, losses["train"])
        statistics.log_statistics(epoch)

        # save model to checkpoint
        roc_auc_score = statistics.get_metric_value("roc_auc", "test", epoch)
        if roc_auc_score > best_roc_auc_score:
            best_roc_auc_score = roc_auc_score

            checkpoint = {
                "experiment_hash": experiment_hash,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "dataset": dataset_name,
                "train_loss": losses["train"],
                "used_fixed_training_triplets": use_fixed_training_triplets,
                "training_type": training_type,
                "alpha": alpha,
                "weight_ones": str(weight_ones),
                "lr": lr,
                "batch_size": train_loader.batch_size,
                "save_dttm": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            checkpoint_path = f"{checkpoints_dir}/{dataset_name}"
            save_checkpoint(checkpoint, checkpoint_path)
    
    # save report
    statistics.save_statistics(f"{checkpoints_dir}/train_report_{experiment_hash}.xlsx")


def train_MSE(model, dataset_name: str, train_loader: DataLoader, 
            test_loader: DataLoader, n_epochs: int, device, checkpoints_dir: str, 
            molecule_augmentator: MoleculeAugmentator = None, lr: float = None):
    
    model = model.to(device)
    experiment_hash = uuid4().hex
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_loss = []
    test_loss = []
    best_loss = sys.float_info.max

    for epoch in range(0, n_epochs):
        
        checkpoint = {}

        for state, loader in zip(["train", "test"], [train_loader, test_loader]):
    
            # calculated parameters
            running_loss = 0.0

            if state == "train":
                model.train()
            else:
                model.eval()

            for batch_id, (mfs, labels, smiles) in enumerate(loader):

                with torch.set_grad_enabled(state == 'train'):

                    if molecule_augmentator and state == "train":
                        mfs = molecule_augmentator.transform_batch(mfs, smiles)

                    mfs, labels = mfs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    outputs = model(mfs)
                    loss = criterion(outputs, labels)

                    if state == "train":
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item()

            epoch_loss = round(running_loss / (batch_id + 1), 5)
            logging.info(f"Epoch: {epoch}, state: {state}, loss: {epoch_loss}")

            # update report
            if state == "train":
                train_loss.append(epoch_loss)
            else:
                test_loss.append(epoch_loss)

        if epoch_loss < best_loss:

            best_loss = epoch_loss

            # save model to checkpoint
            checkpoint["experiment_hash"] = experiment_hash
            checkpoint["epoch"] = epoch
            checkpoint["model_state_dict"] = model.state_dict()
            checkpoint["dataset"] = dataset_name
            checkpoint['train_loss'] = train_loss
            checkpoint['test_loss'] = test_loss
            checkpoint["save_dttm"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            checkpoint_path = f"{checkpoints_dir}/{dataset_name}"
            save_checkpoint(checkpoint, checkpoint_path)
    
    # save report
    report_df = pd.DataFrame({
        "epoch": [n_epoch for n_epoch in range(0, n_epochs)], 
        "train_loss": train_loss, 
        "test_loss": test_loss})
    report_df.to_excel(f"{checkpoints_dir}/train_report_{experiment_hash}.xlsx", index=False)