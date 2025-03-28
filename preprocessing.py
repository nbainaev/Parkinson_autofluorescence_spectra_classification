import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from typing import Union
import os

#Чтение сырых данных с файла

def read_folder(folder_path: str) -> pd.DataFrame:
    raw_data = []
    files = os.listdir(folder_path)
    skip_head = False
    start = 5
    ids = []
    for file in files:
        with open(folder_path + "\\" + file, "r", encoding='utf-16-le') as f:
            print(f"Reading file: {file}.....\n")
            for line in f.readlines()[start:]:
                if "Wavelenghts" in line:
                    if not skip_head:
                        line = list(map(float, line.replace(',', '.').split(sep=";")[1:]))
                        index = line.index(401.733364293274)      # Находим индекс длины волны 401 нм
                        line = line[index:]           # Вырезаем все, что слева от 401 нм
                        columns = line
                        start = 6
                        skip_head = ~skip_head
                else:
                    line = list(map(int, line.split(sep=";")))[1:]
                    line = line[index:]
                    raw_data.append(line)
                    ids.append(file)
            print(f"File {file} has been read correctly")
    raw_data = pd.DataFrame(raw_data, index=ids, columns=columns)                
    return raw_data
    


#Удаление горячих пикселей из спектра

def delete_hot_pixels(arr: list, q=0.9, window = 3):
    buf_arr = pd.Series(np.array([]))
    for i in range(1, len(arr) - 1):
        x1 = arr[i-1]
        x2 = arr[i]
        x3 = arr[i+1]
        buf_arr[i] = abs((x2-x1)*(x3-x2))
    q_value = buf_arr.quantile(q)
    filtered_arr = [arr[0]]
    for i in range(1, len(arr)-1):
        if buf_arr[i] > q_value:
            filtered_arr.append((arr[i-window] + arr[i+window]) / 2)
        else:
            filtered_arr.append(arr[i])
    filtered_arr.append(arr[-1])
    return filtered_arr

# D-нормировка

def d_norm(data: Union[pd.Series, pd.DataFrame]):
    laser_wave = 678.806263180468
    cutoff_wave = 676.874469938485
    if isinstance(data, pd.Series):
        data = data.loc[:cutoff_wave].div(data.loc[laser_wave:].mean())
    else:
        data = data.loc[:, :cutoff_wave].div(data.loc[:, laser_wave:].mean(axis=1), axis=0)
    return data

# I-нормировка

def i_norm(data_dnorm: pd.DataFrame) -> pd.DataFrame:

    mean_values = data_dnorm.mean().to_numpy().reshape(-1, 1)
    reg = LinearRegression()
    data_Inorm_arr = []
    for i in range(len(data_dnorm)):
        x = np.array(list(data_dnorm.iloc[i])).reshape(-1, 1)
        reg.fit(mean_values, x)
        A, B = reg.coef_[0][0], reg.intercept_[0]
        x = (x - B) / A
        data_Inorm_arr.append(x.reshape(-1,))

    data_Inorm = np.asarray(data_Inorm_arr)
    data_Inorm = pd.DataFrame(data_Inorm, columns=data_dnorm.keys(), index=data_dnorm.index)

    data_Inorm.columns = [f"I{i[1:]}" for i in data_Inorm.keys()]
    return data_Inorm