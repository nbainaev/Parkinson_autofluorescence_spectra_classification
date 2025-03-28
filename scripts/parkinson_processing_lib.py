import numpy as np
import pandas as pd
from pathlib import Path
import os
import math
from sklearn.linear_model import LinearRegression

def sigma(x, mean, N):
    y = x.shape[1]
    return (np.sum((x - np.transpose(np.array([list(mean)] * y))) ** 2, axis=1) / N) ** 0.5

def sigma_1(x, mean):
    N = len(mean)
    return (np.sum((x - mean) ** 2, axis=0) / N) ** 0.5

def read_folder(folder_path: Path, group: str, parkinson: int, window_size: int=3):
    raw_data = []
    path_parkinson = sorted(list(map(str, folder_path.glob("*.txt"))))
    name_group_park = []
    all_paths = [(path_parkinson, group, parkinson)]

    relative_fluo = []
    for paths, group, label in all_paths:
        for path in paths:
            file_name = os.path.basename(path)
            with open(path, "r", encoding='utf-16-le') as f:
                for line in f.readlines()[5:]:
                    if "Wavelenghts" in line:
                        if not raw_data:
                            line = list(map(float, line.replace(',', '.').split(sep=";")[1:]))
                            index = line.index(401.733364293274)      # Находим индекс длины волны 401 нм
                            line = line[index:]           # Вырезаем все, что слева от 401 нм
                            name_group_park = [["Name", "Group", "PARKINSON"]]
                            raw_data.append(line)

                    else:
                        line = list(map(int, line.split(sep=";")))[1:]
                        line = line[index:]

                        name_group_park.append([os.path.splitext(file_name)[0], group, label])
                        raw_data.append(line)

    raw_data_head = list(raw_data[0])
    raw_data = np.asarray(raw_data[1:])
    for i in range(window_size // 2, raw_data.shape[1] - window_size // 2):
        mean_red = np.mean(raw_data[:, i - window_size // 2:i + window_size // 2], axis=1)
        sigma_red = np.mean(sigma(raw_data[:, i - window_size // 2:i + window_size // 2 + 1], mean_red, window_size), axis=0)
        delta_minus = np.mean(raw_data[:, i] - raw_data[:, i-1])
        delta_plus = np.mean(raw_data[:, i+1] - raw_data[:, i])
        if abs(delta_minus) > 2 * sigma_red or abs(delta_plus) > 2 * sigma_red:
            raw_data[:, i] = (raw_data[:, i-1] + raw_data[:, i+1]) / 2

    return raw_data, raw_data_head, name_group_park

def moving_avg(x, n = 5):
    ret = np.cumsum(x, dtype=float, axis=1)
    ret[:, n:] = ret[:, n:] - ret[:, :-n]
    return ret[:, n - 1:] / n

def d_norm(raw_data: np.array, raw_data_head: list, name_group_park: list, window_size: int=5) -> pd.DataFrame:
    laser_index = raw_data_head.index(678.806263180468)    # индекс начала лазерного пика
    cutoff_index = raw_data_head.index(676.874469938485)   # индекс отсечки
    laser_mean = np.mean(raw_data[:, laser_index:], axis=1).reshape(-1, 1)
    fluo_mean = np.mean(raw_data[:, :laser_index], axis=1).reshape(-1, 1)
    data_norm = moving_avg(raw_data[:, :cutoff_index + 1] / laser_mean, n=window_size)
    mean_norm = np.mean(data_norm, axis=0)
    sigma_norm = sigma_1(data_norm, mean_norm)
    data_norm_head = raw_data_head[window_size // 2: cutoff_index - window_size // 2 + 1] + name_group_park[0]

    data_Dnorm = pd.DataFrame(np.concatenate((data_norm, np.asarray(name_group_park[1:])), axis=1), columns=data_norm_head)
    data_Dnorm[data_Dnorm.keys()[:-3]] = data_Dnorm[data_Dnorm.keys()[:-3]].astype(float)
    data_Dnorm.columns = [f"D{math.floor(val)}" for val in data_Dnorm.keys()[:-3]] + list(data_Dnorm.keys()[-3:])

    return data_Dnorm

def i_norm(data_dnorm: pd.DataFrame) -> pd.DataFrame:

    mean_values = data_dnorm[data_dnorm.keys()[:-2]].mean().to_numpy().reshape(-1, 1)
    reg = LinearRegression()
    data_Inorm_arr = []
    for i in range(len(data_dnorm['GROUP'])):
        x = np.array(list(data_dnorm.iloc[i])[:-2]).reshape(-1, 1)
        reg.fit(mean_values, x)
        A, B = reg.coef_[0][0], reg.intercept_[0]
        x = (x - B) / A
        data_Inorm_arr.append(x.reshape(-1,))

    data_Inorm= np.asarray(data_Inorm_arr)
    data_Inorm = pd.DataFrame(data_Inorm, columns=data_dnorm.keys()[:-2])
    data_Inorm[['GROUP', 'PARKINSON']] = data_dnorm[['GROUP', 'PARKINSON']]

    data_Inorm.columns = [f"I{i[1:]}" for i in data_Inorm.keys()[:-2]] + ['GROUP', 'PARKINSON']
    data_Inorm['PARKINSON'] = data_Inorm['PARKINSON'].astype(int)
    return data_Inorm

def create_database(control_path: str, parkinson_path: str, comparison_path: str=None) -> pd.DataFrame:
    raw_data_control, raw_data_head_control, name_group_control = read_folder(control_path, "Control", 0)
    raw_data = raw_data_control.copy()
    raw_data_park, raw_data_head_park, name_group_park = read_folder(parkinson_path, "Parkinson", 1)
    raw_data = np.concatenate((raw_data, raw_data_park), axis=0)
    if comparison_path:
        raw_data_comp, raw_data_head_comp, name_group_comp = read_folder(control_path, "Comparison", 0)
        raw_data = np.concatenate((raw_data, raw_data_comp), axis=0)
        raw_data_head = raw_data_head_control + raw_data_head_park + raw_data_head_comp
        name_group = name_group_control + name_group_park[1:] + name_group_comp[1:]
    else:
        raw_data_head = raw_data_head_control + raw_data_head_park
        name_group = name_group_control + name_group_park

    D_norm_data = d_norm(raw_data, raw_data_head, name_group)
    I_norm_data = i_norm(D_norm_data)
    data = pd.concat([D_norm_data[D_norm_data.keys()[:-3]], I_norm_data], axis=1)

    return data