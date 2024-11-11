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
import numpy as np

def train_triplet(model, dataset_name: str, train_loader: DataLoader, test_loader: DataLoader, 
                  n_epochs: int, device, checkpoints_dir: str, use_fixed_training_triplets: bool = False,
                  training_type: str = None, alpha: float = None):
    
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-5)    
    # criterion_triplet_loss = nn.TripletMarginLoss()
    weights_1 = len(train_loader.dataset.indices_0) / len(train_loader.dataset.indices_1)
    criterion_triplet_loss = WeightedTripletMarginLoss(device, train_loader.batch_size, weights_1)
    batch_shaper = BatchShaper(device, training_type, alpha)

    train_loss = []
    test_loss = []

    for epoch in range(0, n_epochs):
        
        checkpoint = {}

        # set fixed training dataset for models comparison
        if epoch > 0 and use_fixed_training_triplets:
                train_loader.dataset.refresh_fixed_triplets(train_loader.dataset.seed_fixed_triplets + epoch)

        for state, loader in zip(["train", "test"], [train_loader, test_loader]):
            
            # calculated parameters
            running_loss = 0.0
            distances_1_1_means = []
            distances_0_0_means = []
            distances_0_1_means = []
            labels_1 = []

            if state == "train":
                model.train()
            else:
                model.eval()

            for batch_id, (anchor_mf, positive_mf, negative_mf, anchor_label) in enumerate(loader):

                with torch.set_grad_enabled(state == 'train'):
                    
                    optimizer.zero_grad()

                    anchor_mf, positive_mf, negative_mf, anchor_label, distances_1_1_mean, distances_0_0_mean, distances_0_1_mean \
                        = batch_shaper.shape_batch(anchor_mf, positive_mf, negative_mf, anchor_label, model, state)

                    loss = criterion_triplet_loss(anchor_mf, positive_mf, negative_mf, anchor_label)
                    
                    distances_1_1_means.append(distances_1_1_mean)
                    distances_0_0_means.append(distances_0_0_mean)
                    distances_0_1_means.append(distances_0_1_mean)
                    labels_1.append(len((anchor_label == 1).nonzero()[:,0].tolist()))

                    if state == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item()

            epoch_loss = round(running_loss / (batch_id + 1), 5)
            epoch_distances_1_1 = round(np.average([m for m in distances_1_1_means if not np.isnan(m)]), 5)
            epoch_distances_0_1 = round(np.average([m for m in distances_0_1_means if not np.isnan(m)]), 5)
            epoch_distances_0_0 = round(np.average([m for m in distances_0_0_means if not np.isnan(m)]), 5)

            logging.info(f"Epoch: {epoch}, state: {state}, loss: {epoch_loss}")
            logging.info(f"distances_1_1: {epoch_distances_1_1}, distances_0_1: {epoch_distances_0_1}, distances_0_0: {epoch_distances_0_0}")
            logging.info(f"min(labels_1): {min(labels_1)}")
            logging.info(f"min(labels_1): {max(labels_1)}")


            # update report
            if state == "train":
                train_loss.append(epoch_loss)
            else:
                test_loss.append(epoch_loss)

        # save model to checkpoint
        checkpoint["epoch"] = epoch
        checkpoint["model_state_dict"] = model.state_dict()
        checkpoint["dataset"] = dataset_name
        checkpoint['train_loss'] = train_loss
        checkpoint['test_loss'] = test_loss
        checkpoint['used_fixed_training_triplets'] = use_fixed_training_triplets
        checkpoint["save_dttm"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        checkpoint_path = f"{checkpoints_dir}/{dataset_name}_{epoch}"
        save_checkpoint(checkpoint, checkpoint_path)
    
    # save report
    report_df = pd.DataFrame({
        "epoch": [n_epoch for n_epoch in range(0, n_epochs)], 
        "train_loss": train_loss, 
        "test_loss": test_loss})
    report_df.to_excel(f"{checkpoints_dir}/train_report_{dataset_name}.xlsx", index=False)


def train_MSE(model, dataset_name: str, train_loader: DataLoader, 
            test_loader: DataLoader, n_epochs: int, device, checkpoints_dir: str):
    
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-5)    
    criterion = nn.MSELoss()

    train_loss = []
    test_loss = []

    for epoch in range(0, n_epochs):
        
        checkpoint = {}

        for state, loader in zip(["train", "test"], [train_loader, test_loader]):
    
            # calculated parameters
            running_loss = 0.0

            if state == "train":
                model.train()
            else:
                model.eval()

            for batch_id, (mfs0, mfs1, targets) in enumerate(loader):

                with torch.set_grad_enabled(state == 'train'):
                    
                    mfs0, mfs1, targets = mfs0.to(device), mfs1.to(device), targets.to(device)
                    optimizer.zero_grad()

                    outputs = model(mfs0, mfs1)
                    loss = criterion(outputs, targets)

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

        # save model to checkpoint
        checkpoint["epoch"] = epoch
        checkpoint["model_state_dict"] = model.state_dict()
        checkpoint["dataset"] = dataset_name
        checkpoint['train_loss'] = train_loss
        checkpoint['test_loss'] = test_loss
        checkpoint["save_dttm"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        checkpoint_path = f"{checkpoints_dir}/{dataset_name}_{epoch}"
        save_checkpoint(checkpoint, checkpoint_path)
    
    # save report
    report_df = pd.DataFrame({
        "epoch": [n_epoch for n_epoch in range(0, n_epochs)], 
        "train_loss": train_loss, 
        "test_loss": test_loss})
    report_df.to_excel(f"{checkpoints_dir}/train_report_{dataset_name}.xlsx", index=False)