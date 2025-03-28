
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from sklearn.metrics import roc_auc_score, accuracy_score

class Discriminator(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim

        self.first_layer = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dims[0]),
            nn.BatchNorm1d(self.hidden_dims[0]),
            nn.LeakyReLU(),
        )

        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
            nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]),
            nn.BatchNorm1d(self.hidden_dims[i + 1]),
            nn.LeakyReLU(),
            ) for i in range(len(self.hidden_dims) - 1)])

        self.last_layer = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], out_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.first_layer(x)
        for layer in self.hidden_layers:
            out = layer(out)
        out = self.last_layer(out)
        return out

def create_default_dataloader(data: pd.DataFrame, labels: pd.Series, test_size: float=0.2, random_state: int=42, batch_size: int=32) -> tuple:
    labels =  torch.tensor(labels.to_numpy())
    data_for_model = torch.tensor(data.to_numpy(), dtype=torch.float)
    data_train, data_val, labels_train, labels_val = train_test_split(data_for_model, labels, test_size=0.25, stratify=labels, random_state=42)
    train_dataset = TensorDataset(data_train, labels_train)
    val_dataset = TensorDataset(data_val, labels_val)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    return train_dataloader, val_dataloader

def create_default_dataset(data: pd.DataFrame, labels: pd.Series, test_size: float=0.2, random_state: int=42, batch_size: int=32) -> tuple:
    features = data.columns[:-3]
    data_train = data[features]
    labels =  torch.tensor(labels.to_numpy())
    data_for_model = torch.tensor(data_train.to_numpy(), dtype=torch.float)
    data_train_, data_val, labels_train, labels_val = train_test_split(data_for_model, labels, test_size=0.25, stratify=labels, random_state=42)
    train_dataset = TensorDataset(data_train_, labels_train)
    val_dataset = TensorDataset(data_val, labels_val)
    return train_dataset, val_dataset

def train(model, epochs, optim, criterion, train_dataloader, val_dataloader, scheduler=None, logging=False, tqdm_desc="Epoch") -> tuple:
    losses_train = []
    losses_val = []
    acc = []
    roc_auc = [[], []]
    device = "cuda" if next(model.parameters()).is_cuda else "cpu"
    for epoch in tqdm(range(epochs), desc=tqdm_desc):
        loss_train = []
        loss_val = []
        model.train()
        for batch, labels in train_dataloader:
            optim.zero_grad()
            batch = batch.to(torch.float)
            labels = labels.to(device)
            logits = model(batch.to(device))

            loss = criterion(logits, labels)
            loss.backward()
            optim.step()

            loss_train.append(loss.detach().cpu().numpy())

        losses_train.append(sum(loss_train) / len(loss_train))

        model.eval()
        preds_epoch = np.array([])
        labels_epoch = np.array([])
        for batch, labels in val_dataloader:
            with torch.no_grad():
                batch = batch.to(torch.float)
                logits = model(batch.to(device))
                preds = torch.argmax(logits, -1).to(torch.float)
                loss = criterion(logits, labels.to(device))
                loss_val.append(loss.detach().cpu().numpy())
                preds_epoch = np.concatenate((preds_epoch, preds.detach().cpu().numpy()))
                labels_epoch = np.concatenate((labels_epoch, labels.detach().cpu().numpy()))

        if scheduler != None:
            scheduler.step()
        losses_val.append(sum(loss_val) / len(loss_val))
        acc.append(accuracy_score(labels_epoch, preds_epoch))

        if len(acc) == 1:
            best_params = model.parameters()
        elif acc[-1] > acc[-2]:
            best_params = model.parameters()

        if logging:
            print(f"Epoch {epoch + 1}:")
            print(f"Train loss: {losses_train[-1]:.2f}, Val loss: {losses_val[-1]:.2f}, Accuracy: {acc[-1]:.2f}, ROC-AUC: {roc_auc_score(labels_epoch, preds_epoch):.2f}\n")

    return losses_train, losses_val, acc, roc_auc, best_params