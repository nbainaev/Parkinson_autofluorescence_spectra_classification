import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LinearRegression
import math
import json
import pickle
from src.model import Discriminator

def sigma(x, mean, N):
    y = x.shape[1]
    return (np.sum((x - np.transpose(np.array([list(mean)] * y))) ** 2, axis=1) / N) ** 0.5

def sigma_1(x, mean):
    N = len(mean)
    return (np.sum((x - mean) ** 2, axis=0) / N) ** 0.5

def moving_avg(x, n = 5):
    ret = np.cumsum(x, dtype=float, axis=1)
    ret[:, n:] = ret[:, n:] - ret[:, :-n]
    return ret[:, n - 1:] / n

def d_norm(raw_data: np.array, raw_data_head: list, name_group_park: list, window_size: int=1) -> pd.DataFrame:
    laser_index = raw_data_head.index(678.806263180468)    # индекс начала лазерного пика
    cutoff_index = raw_data_head.index(676.874469938485)   # индекс отсечки
    laser_mean = np.mean(raw_data[:, laser_index:], axis=1).reshape(-1, 1)
    data_norm = moving_avg(raw_data[:, :cutoff_index + 1] / laser_mean, n=window_size)
    data_norm_head = raw_data_head[window_size // 2: cutoff_index - window_size // 2 + 1] + name_group_park[0]

    data_Dnorm = pd.DataFrame(np.concatenate((data_norm, np.asarray(name_group_park[1:])), axis=1), columns=data_norm_head)
    data_Dnorm[data_Dnorm.keys()[:-3]] = data_Dnorm[data_Dnorm.keys()[:-3]].astype(float)
    data_Dnorm.columns = [f"D{math.floor(val)}" for val in data_Dnorm.keys()[:-3]] + list(data_Dnorm.keys()[-3:])

    return data_Dnorm

def smooth(data: np.ndarray, window_size: int=3) -> np.ndarray:
    raw_data = data.copy()
    for i in range(window_size // 2, raw_data.shape[1] - window_size // 2):
        mean_red = np.mean(raw_data[:, i - window_size // 2:i + window_size // 2], axis=1)
        sigma_red = np.mean(sigma(raw_data[:, i - window_size // 2:i + window_size // 2 + 1], mean_red, window_size), axis=0)
        delta_minus = np.mean(raw_data[:, i] - raw_data[:, i-1])
        delta_plus = np.mean(raw_data[:, i+1] - raw_data[:, i])
        if abs(delta_minus) > 2 * sigma_red or abs(delta_plus) > 2 * sigma_red:
            raw_data[:, i] = (raw_data[:, i-1] + raw_data[:, i+1]) / 2
    return raw_data

def i_norm(data_dnorm: pd.DataFrame) -> pd.DataFrame:

    mean_values = data_dnorm[data_dnorm.keys()[:-3]].mean().to_numpy().reshape(-1, 1)
    reg = LinearRegression()
    data_Inorm_arr = []
    for i in range(len(data_dnorm['Group'])):
        x = np.array(list(data_dnorm.iloc[i])[:-3]).reshape(-1, 1)
        reg.fit(mean_values, x)
        A, B = reg.coef_[0][0], reg.intercept_[0]
        x = (x - B) / A
        data_Inorm_arr.append(x.reshape(-1,))

    data_Inorm= np.asarray(data_Inorm_arr)
    data_Inorm = pd.DataFrame(data_Inorm, columns=data_dnorm.keys()[:-3])
    data_Inorm[['Name', 'Group', 'PARKINSON']] = data_dnorm[data_dnorm.keys()[-3:]]

    data_Inorm.columns = [f"I{i[1:]}" for i in data_Inorm.keys()[:-3]] + ['Name', 'Group', 'PARKINSON']
    data_Inorm['PARKINSON'] = data_Inorm['PARKINSON'].astype(int)
    return data_Inorm


def process(path: str, pca=None, mode: str="train", pca_mode: str="yes", n_components: int=10) -> pd.DataFrame:
    data = pd.read_excel(path, index_col=0)
    n = 3
    if mode == "train":
        data = data[~data["STAGE"].isna()]
    data_model = data.loc[:, 401.733364293274:]
    data_model["Name"] = data["IDcard"]
    data_model["Group"] = data["GROUP"]
    if mode == "train":
        data_model["STAGE"] = data["STAGE"].apply(lambda x: 0 if "stage" not in x.lower() else int(x[0]))
    else:
        data_model["STAGE"] = [0 for i in range(data_model.shape[0])]
    raw_data_numpy = smooth(data_model[data_model.columns[:-n]].to_numpy())
    name_group_park = [data_model.columns[-n:].to_list()] + [data_model.loc[i, data_model.columns[-n:]].to_list() for i in data_model.index]
    d_norm_df = d_norm(raw_data=raw_data_numpy, name_group_park=name_group_park, raw_data_head=data_model.columns[:-n].to_list())
    i_norm_df = i_norm(d_norm_df)
    all_data = pd.concat([d_norm_df.drop(columns=d_norm_df.columns[-n:]), i_norm_df], axis=1)
    if pca_mode == "yes":
        if mode == "train":
            pca = PCA(n_components=n_components)
            data_for_model = pca.fit_transform(all_data[all_data.columns[:-n]])
            data_for_model = pd.DataFrame(data_for_model)
            data_for_model["target"] = all_data["PARKINSON"].to_list()
            del data, data_model, d_norm_df, i_norm_df, all_data
            return data_for_model, pca
        else:
            data_for_model = pca.transform(all_data[all_data.columns[:-n]])
            data_for_model = pd.DataFrame(data_for_model)
            del data, data_model, d_norm_df, i_norm_df, all_data
            return data_for_model
    else:
        data_for_model = d_norm_df[d_norm_df.columns[:-n]].copy()
        buf = []
        for value in all_data["PARKINSON"].to_list():
            buf.append(value if value == 0 else 1)
        data_for_model["target"] = buf
        return data_for_model


def create_default_dataloader(data: pd.DataFrame, labels: pd.Series, test_size: float=0.2, random_state: int=42, batch_size: int=32) -> tuple:
    labels =  torch.tensor(labels.to_numpy())
    data_for_model = torch.tensor(data.to_numpy(), dtype=torch.float)
    data_train, data_val, labels_train, labels_val = train_test_split(data_for_model, labels, test_size=test_size, stratify=labels, random_state=random_state)
    train_dataset = TensorDataset(data_train, labels_train)
    val_dataset = TensorDataset(data_val, labels_val)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    return train_dataloader, val_dataloader

def train(model, epochs, optim, criterion, train_dataloader, val_dataloader, scheduler=None, logging=False, tqdm_desc="Processed") -> tuple:
    losses_train = []
    losses_val = []
    acc = []
    device = "cuda" if next(model.parameters()).is_cuda else "cpu"
    for epoch in tqdm(range(epochs), desc=tqdm_desc):
        loss_train = []
        loss_val = []
        model.train()
        for batch, labels in train_dataloader:
            optim.zero_grad()
            batch = batch.to(torch.float)
            labels = labels.to(device)
            logits = model(batch.to(device)).squeeze().to(torch.float)

            loss = criterion(logits, labels.to(torch.float))
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
                logits = model(batch.to(device)).squeeze().to(torch.float)
                preds = (logits > 0.5).to(torch.int)
                loss = criterion(logits, labels.to(torch.float).to(device))
                loss_val.append(loss.detach().cpu().numpy())
                preds_epoch = np.concatenate((preds_epoch, preds.detach().cpu().numpy()))
                labels_epoch = np.concatenate((labels_epoch, labels.detach().cpu().numpy()))

        if scheduler != None:
            scheduler.step()
        losses_val.append(sum(loss_val) / len(loss_val))
        acc.append(accuracy_score(labels_epoch, preds_epoch))

        if len(acc) == 1:
            best_params = model.state_dict()
        elif acc[-1] > acc[-2]:
            best_params = model.state_dict()

        if logging:
            print(f"Epoch {epoch + 1}:")
            print(f"Train loss: {losses_train[-1]:.2f}, Val loss: {losses_val[-1]:.2f}, Accuracy: {acc[-1]:.2f}\n")
    print(f"Training complete successfully. Best accuracy on validation: {max(acc):.4f}")
    return losses_train, losses_val, acc, best_params

def predict(model, data):
    device = "cuda" if next(model.parameters()).is_cuda else "cpu"
    model.to(device)
    model.eval()
    preds_epoch = np.array([])
    with torch.no_grad():
        for i in tqdm(data.index, desc="Processed"):
            spec = data.iloc[i]
            spec = torch.tensor(spec).to(torch.float).unsqueeze(0)
            logits = model(spec.to(device))
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, -1).to(torch.float)
            preds_epoch = np.concatenate((preds_epoch, preds.detach().cpu().numpy()))

    return preds_epoch


if __name__ == "__main__":
    force_train = 0
    if os.path.isfile("params.json"):
        with open("params.json", "r") as file_:
            parameters = json.load(file_)
    else:
        parameters = {
            "path_to_model": "model.pkl", 
            "path_to_pca": "pca.pkl", 
            "path_to_data": "data.xlsx", 
            "in_dim": 10,
            "out_dim": 1,
            "hidden_dims": [64, 16],
            "learning_rate": 5e-4,
            "training epochs": 10,
            "batch_size": 32,
            "test_size": 0.2
            }
    if not os.path.isfile(parameters["path_to_model"]):
        path_to_model = input("The model isn't found. Enter the path to the model (if is doesn't exist, write \"no\"): ")
        if path_to_model == "no":
            force_train = 1
        else:
            parameters["path_to_model"] = path_to_model
            with open(path_to_model, 'rb') as file_:
                model = pickle.load(file_)
    else:
        with open(parameters["path_to_model"], 'rb') as file_:
            model = pickle.load(file_)
    if not os.path.isfile(parameters["path_to_pca"]):
        path_to_pca = input("PCA isn't found. Enter the path to the PCA preprocessor (if is doesn't exist, write \"no\"): ")
        if path_to_pca == "no":
            pass
        else:
            parameters["path_to_pca"] = path_to_pca
            with open(path_to_pca, 'rb') as file_:
                pca = pickle.load(file_)
    else:
        with open(parameters["path_to_pca"], 'rb') as file_:
            pca = pickle.load(file_)
    command = ""
    while command != "stop":
        if force_train == 1:
            print("Only training option is available, starting training the model\n")
            command = "train"
            force_train = 0
        else:
            print("")
            command = input("To train model on data: train \nTo make a prediction: predict \nTo exit: stop \nTo show parameters: show params \nTo change parameters: change params \nWrite your command here: ")
        if command == "train":
            print("")
            if not os.path.isfile(parameters["path_to_data"]):
                parameters["path_to_data"] = input("Data is not found. Enter the path to the data: ")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            pca_mode = input("Do you want to train with PCA (yes/no): ")
            if pca_mode == "yes":
                data, pca = process(parameters["path_to_data"], parameters["in_dim"], pca_mode=pca_mode)
            else:
                data = process(parameters["path_to_data"], parameters["in_dim"], pca_mode=pca_mode)
            train_loader, val_loader = create_default_dataloader(data.drop(columns=["target"]), data["target"], batch_size=parameters["batch_size"], test_size=parameters["test_size"])
            model = Discriminator(len(data.columns)-1, parameters["hidden_dims"], parameters["out_dim"]).to(device)
            optim = torch.optim.Adam(model.parameters(), lr = parameters["learning_rate"])
            criterion = nn.BCELoss()
            print("Preporation complete, training a model\n")
            losses_train, losses_val, acc, best_params = train(model, parameters["training epochs"], optim, criterion, train_loader, val_loader)

            command = input("Do you wish to save results and best parametes? (yes/no):  ")
            if command == "yes":
                df = pd.DataFrame(np.array([losses_train, losses_val, acc]).T, columns=["Loss_train", "Loss_val", "Accuracy"])
                name = input("Type a name of csv file for saving history of training: ")
                df.to_csv(name + ".csv")
                with open('model.pkl', 'wb') as file_:
                    pickle.dump(model, file_)
                if pca_mode == "yes":
                    with open('pca.pkl', 'wb') as file_:
                        pickle.dump(pca, file_)
                print("All data saved successfully\n")
                print("")
            else:
                print("")
                continue
        elif command == "predict":
            print("")
            path_to_predict = input("Please, enter the path to prediction data(.xlsx): ")
            data = process(path_to_predict, pca, mode="test")
            print("Preporation complete, making a prediction\n")
            labels = predict(model, data)
            print("Prediction complete\n")
            name = input("Type a filename to save result: ")
            pd.DataFrame(labels, columns=["Predicted stage"]).to_csv(name + ".csv")
            print("All predictions saved\n")
            print("")
        elif command == "stop":
            with open("params.json", "w") as file_:
                json.dump(parameters, file_)
                break
        elif command == "show params":
            print("")
            print("{")
            for key, value in parameters.items():
                print(f"{key}: {value}")
            print("}")
        elif command == "change params":
            print("\nCurrent parameters: {")
            for key, value in parameters.items():
                print(f"{key}: {value}")
            print("}\n")
            param = input("Enter, what parameter you want to change: ")
            if type(parameters[param]) == int:
                value = input(f"Enter the new {param}: ")
                parameters[param] = int(value)
            elif type(parameters[param]) == float:
                value = input(f"Enter the new {param}: ")
                parameters[param] = float(value)
            elif type(parameters[param]) == list:
                value = list(map(int, input(f"Enter the new {param} (write dims, separated with coma): ").split(sep=",")))
                parameters[param] = value
            else:
                parameters[param] = value
            print(f"Parameter {param} changed successfully")
        else:
            print("")
            print("Unknown command, please, retry\n")
            print("")