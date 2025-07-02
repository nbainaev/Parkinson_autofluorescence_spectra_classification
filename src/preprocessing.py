import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from typing import Union
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.optimize import least_squares

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
            for i, line in enumerate(f.readlines()[start:]):
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
                    ids.append(file.replace(".txt", f"_{i}"))
            print(f"File {file} has been read correctly")
    raw_data = pd.DataFrame(raw_data, index=ids, columns=columns)                
    return raw_data

def read_folder_new(folder_path: str) -> pd.DataFrame:
    raw_data = []
    files = os.listdir(folder_path)
    files_ = []
    for file in files:
        if '.txt' in file:
            files_.append(file)
    files = files_
    skip_head = False
    start = 5
    ids = []
    for file in files:
        with open(folder_path + "\\" + file, "r") as f:
            print(f"Reading file: {file}.....\n")
            for i, line in enumerate(f.readlines()[start:]):
                if "Wavelenghts" in line:
                    if not skip_head:
                        line = list(map(float, line.replace(',', '.').split('\t')[1:]))
                        columns = line
                        start = 6
                        skip_head = not skip_head
                else:
                    line = list(map(int, line.split(sep="\t")))[1:]
                    raw_data.append(line)
                    ids.append(file.replace(".txt", f"_{i}"))
            print(f"File {file} has been read correctly")
    raw_data = pd.DataFrame(raw_data, index=ids, columns=columns)                
    return raw_data

def interpolate_spectrum(wavelengths: np.ndarray, intensities: np.ndarray, reference_wavelengths: np.ndarray) -> np.ndarray:
    """
    Выполняет сплайн-интерполяцию спектра к референсным длинам волн.
    
    Параметры:
    wavelengths (array-like): Длины волн исходного спектра
    intensities (array-like): Интенсивности исходного спектра
    reference_wavelengths (array-like): Референсные длины волн (150 точек)
    
    Возвращает:
    numpy.ndarray: Интенсивности интерполированного спектра
    """
    # Создаем интерполяционную функцию (кубический сплайн по умолчанию)
    interp_func = interp1d(wavelengths, intensities, 
                           kind='cubic', 
                           bounds_error=False, 
                           fill_value='extrapolate')
    
    # Интерполируем интенсивности на референсные длины волн
    interpolated_intensities = interp_func(reference_wavelengths)
    
    return interpolated_intensities

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from tqdm import tqdm  # для прогресс-бара (опционально)

def batch_piecewise_lsq_interpolation(original_wavelengths: np.ndarray,
                                    original_spectra: np.ndarray,
                                    new_wavelengths: np.ndarray,
                                    window_size: int = 5,
                                    poly_order: int = 2,
                                    n_jobs: int = -1) -> np.ndarray:
    """
    Выполняет кусочную интерполяцию методом МНК для набора спектров.
    
    Параметры:
    ----------
    original_wavelengths : np.ndarray, 1D
        Исходные длины волн (общие для всех спектров).
    original_spectra : np.ndarray, 2D (n_spectra, n_points)
        Массив спектров (каждая строка - отдельный спектр).
    new_wavelengths : np.ndarray, 1D
        Новые длины волн для интерполяции.
    window_size : int, optional
        Количество ближайших точек для МНК (по умолчанию 5).
    poly_order : int, optional
        Порядок полинома (по умолчанию 2).
    n_jobs : int, optional
        Количество ядер для параллельных вычислений (-1 = все ядра).
    
    Возвращает:
    -------
    np.ndarray, 2D (n_spectra, n_new_points)
        Интерполированные спектры.
    """
    # Проверки входных данных
    assert original_spectra.shape[1] == len(original_wavelengths), \
        "Количество точек в спектрах должно совпадать с длиной original_wavelengths"
    
    # Построение KD-дерева для быстрого поиска ближайших точек
    tree = cKDTree(original_wavelengths.reshape(-1, 1))
    
    # Находим ближайшие точки для всех новых длин волн сразу
    _, nearest_indices = tree.query(new_wavelengths.reshape(-1, 1), 
                                  k=window_size)
    
    # Предварительное выделение памяти
    n_spectra = original_spectra.shape[0]
    n_new_points = len(new_wavelengths)
    interpolated_spectra = np.zeros((n_spectra, n_new_points))
    
    # Оптимизированная функция для одного спектра
    def interpolate_single_spectrum(spectrum):
        result = np.zeros(n_new_points)
        for i, idxs in enumerate(nearest_indices):
            x_window = original_wavelengths[idxs]
            y_window = spectrum[idxs]
            
            # Центрирование данных
            x_centered = x_window - x_window.mean()
            
            # МНК аппроксимация
            coeffs = np.polyfit(x_centered, y_window, poly_order)
            x_new_centered = new_wavelengths[i] - x_window.mean()
            result[i] = np.polyval(coeffs, x_new_centered)
        
        return np.clip(result, 0, None)
    
    # Параллельная обработка спектров
    from joblib import Parallel, delayed
    interpolated_spectra = np.array(
        Parallel(n_jobs=n_jobs)(
            delayed(interpolate_single_spectrum)(original_spectra[i]) 
            for i in tqdm(range(n_spectra), desc="Обработка спектров")
        )
    )
    
    return interpolated_spectra





def smooth(data: pd.DataFrame, window_length:int=7, polyorder:int=3) -> pd.DataFrame:
    data_sm = None
    for y in data.to_numpy():
        y_smooth = savgol_filter(y, window_length, polyorder)
        if data_sm is None:
            data_sm = y_smooth
        else:
            data_sm = np.vstack((data_sm, y_smooth))
    df_sm = pd.DataFrame(data_sm, index=data.index, columns=data.columns)
    return df_sm

#Удаление горячих пикселей из спектра

def delete_hot_pixels(arr: list, q: float=0.9,
                       window: int = 3) -> list:
    delta = pd.Series(np.array([]))
    for i in range(1, len(arr) - 1):
        x1 = arr[i-1]
        x2 = arr[i]
        x3 = arr[i+1]
        delta[i] = abs((x2-x1)*(x3-x2))
    q_value = delta.quantile(q)
    filtered_arr = [arr[0]]
    for i in range(1, len(arr)-1):
        if delta[i] > q_value:
            filtered_arr.append(
                sum(arr[i-window:i] + arr[i+1:i+window]) / (window-1)
                )
        else:
            filtered_arr.append(arr[i])
    filtered_arr.append(arr[-1])
    return filtered_arr

# D-нормировка

def d_norm(data: Union[pd.Series, pd.DataFrame],
                laser_wave_left: float = 731.195121105382,
                laser_wave_right: float = 772.244519389919,
                cutoff_wave: float = 676.874469938485):
    if isinstance(data, pd.Series):
        data = data.loc[:cutoff_wave].div(
            data.loc[laser_wave_left:laser_wave_right].mean())
    else:
        data = data.loc[:, :cutoff_wave].div(
            data.loc[:, laser_wave_left:laser_wave_right].mean(axis=1),
            axis=0)
        data.rename(axis=1,
                    mapper=lambda x: f"D{int(x)}",
                    inplace=True)
    return data

# I-нормировка

def i_norm(data_dnorm: pd.DataFrame) -> pd.DataFrame:
    data_Inorm_arr = []
    reg = LinearRegression()
    for group in data_dnorm.index.get_level_values("GROUP").unique():
        data_buf = data_dnorm[data_dnorm.index.get_level_values("GROUP").isin([group])]
        mean_values = data_buf.mean().to_numpy().reshape(-1, 1)
        for i in range(len(data_buf)):
            x = np.array(list(data_buf.iloc[i])).reshape(-1, 1)
            reg.fit(mean_values, x)
            A, B = reg.coef_[0][0], reg.intercept_[0]
            x = (x - B) / A
            data_Inorm_arr.append(x.reshape(-1,))

    data_Inorm = np.asarray(data_Inorm_arr)
    data_Inorm = pd.DataFrame(data_Inorm, columns=data_dnorm.keys(), index=data_dnorm.index)

    data_Inorm.columns = [f"I{i[1:]}" if i[0] == "D" else f"I{i}" for i in data_Inorm.keys()]
    return data_Inorm