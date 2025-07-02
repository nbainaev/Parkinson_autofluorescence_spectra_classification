import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import wandb

from tqdm import tqdm
from sklearn.metrics import precision_score, accuracy_score, recall_score, roc_auc_score

from sklearn.calibration import calibration_curve
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
                                    calib_size: float=0.2,
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
            if self.train_conf["calibration"]:
                y_train, y_calib = train_test_split(y_train, stratify=y_train, test_size=calib_size, random_state=random_state)
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
            
            if self.train_conf["calibration"]:
                X_calib = X.loc[idx[:, y_calib.index, :]]
                y_calib = y.loc[idx[:, y_calib.index, :]]
                labels_calib = torch.tensor(y_calib.to_numpy())
                data_calib = torch.tensor(X_calib.to_numpy(), dtype=torch.float)
                calib_dataset = TensorDataset(data_calib, labels_calib)
                calib_dataloader = DataLoader(calib_dataset, batch_size=batch_size)
            return {"train": train_dataloader, "val": val_dataloader,
                    "calib_data": calib_dataloader, "y_calib": y_calib.to_numpy(), "weight": weight}


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

        def calibrate(self):
            device = "cuda" if next(self.model.parameters()).is_cuda else "cpu"
            calib_dataloader = self.split["calib_data"]
            calib_y = self.split["y_calib"]
            logits = torch.tensor(np.array([]))
            for batch in calib_dataloader:
                if type(batch) == list:
                    batch = batch[0]
                with torch.no_grad():
                    batch = batch.to(torch.float) 
                    logits = torch.concatenate((logits, self.model(batch.to(device)).squeeze().to(torch.float)))
            self.model.calibrator.fit(torch.sigmoid(logits).detach().cpu().numpy(), calib_y)
        
        def plot_calibration_curve(self, y_true, probs, title, linestyle='-'):
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, probs, n_bins=10, strategy='quantile'
            )
            plt.plot(
                mean_predicted_value, 
                fraction_of_positives, 
                linestyle=linestyle, 
                label=title
            )
            plt.plot([0, 1], [0, 1], "k--", label="Идеально калиброванная модель")
       
        def predict(self, pred_data):
            device = "cuda" if next(self.model.parameters()).is_cuda else "cpu"
            self.model.to(device)
            self.model.eval()
            preds_all = np.array([])
            proba_all = np.array([])
            logits_all = np.array([])
            for batch in pred_data:
                if type(batch) == list:
                    batch = batch[0]
                with torch.no_grad():
                    batch = batch.to(torch.float)
                    logits = torch.sigmoid(self.model(batch.to(device)).squeeze().to(torch.float))
                    
                    if self.model_conf["calibrator"] is not None:
                        proba = self.model.calibrator.transform(logits.detach().cpu().numpy())
                        preds = (proba > 0.5).astype(np.int16)
                        preds_all = np.concatenate((preds_all, preds))
                        proba_all = np.concatenate((proba_all, proba))
                        logits_all = np.concatenate((logits_all, logits.detach().cpu().numpy()))
                    else:
                        proba = torch.sigmoid(logits)
                        preds = (logits > 0.5).to(torch.int)
                        preds_all = np.concatenate((preds_all, preds.detach().cpu().numpy()))
                        proba_all = np.concatenate((proba_all, proba.detach().cpu().numpy()))
                        
            return preds_all, proba_all, logits_all
        
        def run(self):
            self.train()
            if self.train_conf["calibration"]:
                self.calibrate()
                preds, proba, logits = self.predict(self.split["val"])
                old_preds = (logits > 0.5).astype(np.int16)
                y_test = np.array([])
                for _, labels in self.split["val"]:
                    y_test = np.concatenate((y_test, labels.detach().cpu().numpy()))
                fig = plt.figure(figsize=(10, 8))
                plt.subplot(2, 1, 1)
                self.plot_calibration_curve(y_test, logits, "До калибровки")
                plt.ylabel("Доля положительных классов")
                plt.title("Калибровочные кривые")
                plt.legend()
   
                plt.subplot(2, 1, 2)
                self.plot_calibration_curve(y_test, proba, "После калибровки")
                plt.xlabel("Средние предсказанные значения")
                plt.ylabel("Доля положительных классов")
                plt.legend()

                plt.tight_layout()
                
                print(f"Accuracy: {accuracy_score(y_test, preds):.2f}")
                print(f"Precision: {precision_score(y_test, preds):.2f}")
                print(f"Recall: {recall_score(y_test, preds):.2f}")
                print(f"ROC-AUC: {roc_auc_score(y_test, logits):.2f}")

                print(f"Accuracy calibrated: {accuracy_score(y_test, old_preds):.2f}")
                print(f"Precision calibrated: {precision_score(y_test, old_preds):.2f}")
                print(f"Recall calibrated: {recall_score(y_test, old_preds):.2f}")
                print(f"ROC-AUC calibrated: {roc_auc_score(y_test, proba):.2f}")
                
                self.logger.log(
                        {
                            'calibration_curve': wandb.Image(fig)
                        }
                    )
                plt.close('all')
