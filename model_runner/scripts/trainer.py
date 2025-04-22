import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt


from tqdm import tqdm
from sklearn.metrics import precision_score, accuracy_score, recall_score, roc_auc_score

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from improved_model import ImprovedDiscriminator


class DiscriminatorTrainer():
        def __init__(self, logger, conf):
            self.logger = logger
            self.train_conf = conf["train"]
            self.data_conf = conf["data"]
            self.model_conf = conf["model"]
            self.epochs = conf["train"]["epochs"]
            self.data = pd.read_excel(conf["run"]["data_path"], index_col=[0, 1, 2])
            self.log_update_rate = conf["run"]["log_update_rate"]
            
            if conf["run"]["split"] == "i_norm":
                X = self.data.loc[:, "I401":"I676"].copy()
                y = pd.Series([1 if x == "Parkinson" else 0 for x in self.data.index.get_level_values("GROUP").to_list()], index = self.data.index)
            elif conf["run"]["split"] == "d_norm":
                X = self.data.loc[:, "D401":"D676"].copy()
                y = pd.Series([1 if x == "Parkinson" else 0 for x in self.data.index.get_level_values("GROUP").to_list()], index = self.data.index)
            else:   
                X = self.data.copy()
                y = pd.Series([1 if x == "Parkinson" else 0 for x in self.data.index.get_level_values("GROUP").to_list()], index = self.data.index)
            
            self.split = self.create_default_dataloader(X, y, **self.data_conf)
            
            self.model_conf["in_dim"] = len(X.columns)
            self.model = ImprovedDiscriminator(**self.model_conf)
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.train_conf["lr"],  # Уменьшенный LR
                #weight_decay=1e-6  # Меньше регуляризации
            )

            # Изменяем шедулер
            if self.train_conf["scheduler"]:
                self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, gamma=self.train_conf["gamma"],
                                                            step_size=self.train_conf["step_size"])
            else:
                self.scheduler = None

            # Функция потерь
            self.criterion = nn.BCEWithLogitsLoss(weight=torch.tensor([self.split["weight"]]))
            
        def create_default_dataloader(self,
                                    X: pd.DataFrame,
                                    y: pd.Series,
                                    test_size: float=0.2,
                                    random_state: int=42,
                                    batch_size: int=32,
                                    n_components=50,
                                    pca_on=True) -> tuple:
            idx = pd.IndexSlice
            if pca_on:
                pca = PCA(n_components)
                X = pd.DataFrame(pca.fit_transform(X), index=X.index)
            y_buf = y.groupby(level=1).max()
            y_train, y_val = train_test_split(y_buf, stratify=y_buf, test_size=test_size, random_state=random_state)
            X_train = X.loc[idx[:, y_train.index, :]]
            X_val = X.loc[idx[:, y_val.index, :]]
            y_train = y.loc[idx[:, y_train.index, :]]
            y_val = y.loc[idx[:, y_val.index, :]]
            weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

            labels_train = torch.tensor(y_train.to_numpy())
            labels_val = torch.tensor(y_val.to_numpy())
            data_train = torch.tensor(X_train.to_numpy(), dtype=torch.float)
            data_val = torch.tensor(X_val.to_numpy(), dtype=torch.float)
            
            train_dataset = TensorDataset(data_train, labels_train)
            val_dataset = TensorDataset(data_val, labels_val)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
            return {"train": train_dataloader, "val": val_dataloader, "weight": weight}


        def train(self, tqdm_desc="Processed") -> None:
            train_dataloader = self.split["train"]
            val_dataloader = self.split["val"]
            losses_train = []
            losses_val = []
            acc = []
            precision = []
            recall = []
            auc = []
            device = "cuda" if next(self.model.parameters()).is_cuda else "cpu"
            for epoch in tqdm(range(1, self.epochs+1), desc=tqdm_desc):
                loss_train = []
                loss_val = []
                self.model.train()
                for batch, labels in train_dataloader:
                    self.optimizer.zero_grad()
                    batch = batch.to(torch.float)
                    labels = labels.to(device)
                    logits = self.model(batch.to(device)).squeeze().to(torch.float)

                    loss = self.criterion(logits, labels.to(torch.float))
                    loss.backward()
                    self.optimizer.step()

                    loss_train.append(loss.detach().cpu().numpy())

                losses_train.append(sum(loss_train) / len(loss_train))

                self.model.eval()
                preds_epoch = np.array([])
                labels_epoch = np.array([])
                probs_epoch = np.array([])
                for batch, labels in val_dataloader:
                    with torch.no_grad():
                        batch = batch.to(torch.float)
                        logits = self.model(batch.to(device)).squeeze().to(torch.float)
                        preds = (logits > 0.5).to(torch.int)
                        loss = self.criterion(logits, labels.to(torch.float).to(device))
                        loss_val.append(loss.detach().cpu().numpy())
                        preds_epoch = np.concatenate((preds_epoch, preds.detach().cpu().numpy()))
                        labels_epoch = np.concatenate((labels_epoch, labels.detach().cpu().numpy()))
                        probs_epoch = np.concatenate((probs_epoch, logits))


                if self.scheduler != None:
                    self.scheduler.step()
                # if len(acc) == 1:
                #     best_params = self.model.state_dict()
                # elif acc[-1] > acc[-2]:
                #     best_params = self.model.state_dict()

                
                if self.logger is not None:
                    acc.append(accuracy_score(labels_epoch, preds_epoch))
                    losses_val.append(sum(loss_val) / len(loss_val))
                    precision.append(precision_score(labels_epoch, preds_epoch))
                    recall.append(recall_score(labels_epoch, preds_epoch))
                    auc.append(roc_auc_score(labels_epoch, probs_epoch))
                    self.logger.log(
                        {
                            'main_metrics/val_loss': losses_val[-1],
                            'main_metrics/accuracy': acc[-1],
                            'main_metrics/precision': precision[-1],
                            'main_metrics/recall': recall[-1],
                            'main_metrics/roc_auc': auc[-1],
                        }, step=epoch
                    )
        
        def predict(self, model, val_dataloader):
            device = "cuda" if next(model.parameters()).is_cuda else "cpu"
            model.to(device)
            model.eval()
            preds_all = np.array([])
            proba_all = np.array([])
            labels_all = np.array([])
            for batch, labels in val_dataloader:
                with torch.no_grad():
                    batch = batch.to(torch.float)
                    logits = model(batch.to(device)).squeeze().to(torch.float)
                    preds = (logits > 0.5).to(torch.int)
                    proba = torch.sigmoid(logits)
                    preds_all = np.concatenate((preds_all, preds.detach().cpu().numpy()))
                    labels_all = np.concatenate((labels_all, labels.detach().cpu().numpy()))
                    proba_all = np.concatenate((proba_all, proba.detach().cpu().numpy()))

            return preds_all, proba_all, labels_all
