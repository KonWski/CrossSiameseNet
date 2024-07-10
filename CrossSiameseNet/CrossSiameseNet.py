import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
import logging
from datetime import datetime
import pandas as pd
from typing import List

class CrossSiameseNet(nn.Module):
    '''Siamese network using features from other siamese networks'''

    def __init__(self, models: List[nn.Module]):

        super().__init__()

        self.models = models
        self.n_models = len(models)
        self.cf_size = models[0].cf_size
        self.features = nn.Sequential(
            nn.Conv1d(2, 1, 1),
            nn.Linear(2*self.cf_size, self.cf_size),
            nn.BatchNorm1d(self.cf_size)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(2*self.cf_size, self.cf_size),
            nn.BatchNorm1d(self.cf_size),
            nn.Linear(self.cf_size, 1)
        )

        # turn off grads in all parameters 
        for model in self.models:

            model.eval()
            for param in model.parameters():
                param.requires_grad = False

        # initialize the weights
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0.01)
        
        for layer in self.features:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0.01)

    def forward_once(self, x):

        # features collected across all models
        features_submodels = [model.forward_once(x) for model in self.models]
        features_submodels = torch.stack(features_submodels, dim=-2)

        features = self.features(features_submodels)
        
        return features

    def forward(self, mol0, mol1):

        # process two molecules
        features0 = self.forward_once(mol0)
        features1 = self.forward_once(mol1)

        # combine both feature vectors
        features = torch.concat([features0, features1], dim=-1)

        # final output
        output = self.fc(features)

        return output


def save_checkpoint(checkpoint: dict, checkpoint_path: str):
    '''
    saves checkpoint on given checkpoint_path
    '''
    torch.save(checkpoint, checkpoint_path)

    logging.info(8*"-")
    logging.info(f"Saved model to checkpoint: {checkpoint_path}")
    logging.info(f"Epoch: {checkpoint['epoch']}")
    logging.info(8*"-")


def train(model: CrossSiameseNet, train_loader: DataLoader, test_loader: DataLoader, 
            n_epochs: int, device, checkpoints_dir: str):
    
    model = model.to(device)
    optimizer = Adam([param for param in model.fc.parameters()] + [param for param in model.features.parameters()], lr=1e-5)
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
        checkpoint['train_loss'] = train_loss
        checkpoint['test_loss'] = test_loss
        checkpoint["save_dttm"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        checkpoint_path = f"{checkpoints_dir}/CrossSiameseNet_{epoch}"
        save_checkpoint(checkpoint, checkpoint_path)
    
    # save report
    report_df = pd.DataFrame({
        "epoch": [n_epoch for n_epoch in range(0, n_epochs)], 
        "train_loss": train_loss, 
        "test_loss": test_loss})
    report_df.to_excel(f"{checkpoints_dir}/train_report.xlsx", index=False)
