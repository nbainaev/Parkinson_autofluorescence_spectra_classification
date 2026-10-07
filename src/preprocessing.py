import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from typing import Union, Any
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.optimize import least_squares

#Чтение сырых данных с файла
def read_folder(
        folder_path: str, 
        filenames: list | None=None, 
        start_file: str| None=None, 
        stop_file: str | None=None) -> pd.DataFrame:
    raw_data = []
    if filenames is not None:
        files = filenames
    elif start_file is not None:
        files = os.listdir(folder_path)
        start_ind = files.index(start_file)
        stop_ind = files.index(stop_file) + 1 if stop_file is not None else len(files)
        files = files[start_ind:stop_ind]
    else:
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
                        skip_head = not skip_head
                else:
                    line = list(map(int, line.split(sep=";")))[1:]
                    line = line[index:]
                    raw_data.append(line)
                    ids.append(file.replace(".txt", f"_{i + 1}"))
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

def delete_hot_pixels(arr: Union[list, np.ndarray, Any], q: float=0.9,
                      window: int = 3) -> list:
    delta = []
    max_ = len(arr)
    for i in range(1, len(arr) - 1):
        x1 = arr[i-1]
        x2 = arr[i]
        x3 = arr[i+1]
        delta.append((x2-x1)*(x3-x2))
    q_value = np.quantile(delta, 1-q)
    filtered_arr = [arr[0]]
    for i in range(1, len(arr)-1):
        if delta[i] <= q_value:
            filtered_arr.append(
                sum(arr[max(i-window//2, 0):i] +\
                    arr[i+1:min(i+window//2+1, max_)]) / (window-1)
                )
        else:
            filtered_arr.append(arr[i])
    filtered_arr.append(arr[-1])
    return filtered_arr

def delete_hot_pixels_fast(arr: list, q: float = 0.9,
                                     window: int = 3) -> np.ndarray:
    """
    Векторизованная версия с обработкой отрицательных значений.
    """
    arr = np.asarray(arr, dtype=np.float64)
    n = len(arr)
    
    if n < 3:
        return arr.copy()
    
    # 1. Находим отрицательные значения с неотрицательными соседями
    negative_mask = arr < 0
    
    # Создаем маски для соседей
    left_neighbors_positive = np.ones(n, dtype=bool)
    right_neighbors_positive = np.ones(n, dtype=bool)
    
    left_neighbors_positive[1:] = arr[:-1] >= 0  # Левые соседи (кроме первого элемента)
    right_neighbors_positive[:-1] = arr[1:] >= 0  # Правые соседи (кроме последнего элемента)
    
    # Отрицательные значения с неотрицательными соседями
    valid_negative_mask = negative_mask & left_neighbors_positive & right_neighbors_positive
    
    # 2. Находим горячие пиксели
    delta = (arr[1:-1] - arr[:-2]) * (arr[2:] - arr[1:-1])
    q_value = np.quantile(delta, 1 - q)
    hot_mask_center = delta <= q_value
    
    # Расширяем маску горячих пикселей на полный размер массива
    hot_mask = np.zeros(n, dtype=bool)
    hot_mask[1:-1] = hot_mask_center
    
    # 3. Объединяем маски
    process_mask = valid_negative_mask | hot_mask
    process_indices = np.where(process_mask)[0]
    
    if len(process_indices) == 0:
        return arr.copy()
    
    # 4. Векторизованная обработка
    half_window = window // 2
    result = arr.copy()
    
    # Создаем матрицу индексов для всех обрабатываемых точек
    indices_matrix = process_indices[:, None] + np.arange(-half_window, half_window + 1)
    indices_matrix = np.clip(indices_matrix, 0, n - 1)
    
    # Получаем значения в окнах
    window_values = arr[indices_matrix]
    
    # Создаем маску для исключения центральных точек
    center_mask = np.ones(window, dtype=bool)
    center_mask[half_window] = False
    
    # Вычисляем средние значения
    neighbor_values = window_values[:, center_mask]
    
    # Для каждого окна используем только неотрицательных соседей (если есть)
    means = np.zeros(len(process_indices))
    for i in range(len(process_indices)):
        valid_neighbors = neighbor_values[i][neighbor_values[i] >= 0]
        if len(valid_neighbors) > 0:
            means[i] = np.mean(valid_neighbors)
        else:
            means[i] = np.mean(neighbor_values[i])  # если все соседи отрицательные
    
    result[process_indices] = means
    
    return result

def delete_hot_pixels_adaptive(spectrum, window_size=5, sensitivity=2.0):
    """
    Адаптивный метод удаления горячих пикселей на основе локальных градиентов.
    
    Parameters:
    -----------
    spectrum : array-like
        Входной спектр
    window_size : int
        Размер окна для анализа
    sensitivity : float
        Чувствительность обнаружения (1.0 - низкая, 3.0 - высокая)
    
    Returns:
    --------
    cleaned_spectrum : numpy.ndarray
        Очищенный спектр
    """
    spectrum = np.array(spectrum, dtype=np.float32)
    n = len(spectrum)
    cleaned_spectrum = spectrum.copy()
    
    window_size = max(3, window_size if window_size % 2 == 1 else window_size + 1)
    half_window = window_size // 2
    
    for i in range(half_window, n - half_window):
        window = spectrum[i-half_window:i+half_window+1]
        
        # Рассчитываем градиенты слева и справа
        left_gradient = abs(spectrum[i] - spectrum[i-1])
        right_gradient = abs(spectrum[i] - spectrum[i+1])
        
        # Рассчитываем ожидаемый градиент на основе соседей
        expected_gradient = 0.5 * (abs(spectrum[i-1] - spectrum[i-2]) + 
                                 abs(spectrum[i+1] - spectrum[i+2]))

        # Если текущие градиенты значительно превышают ожидаемые
        if (left_gradient > sensitivity * expected_gradient and 
            right_gradient > sensitivity * expected_gradient and
            left_gradient * right_gradient > 0):  # Оба градиента в одном направлении
            
            # Проверяем, что это действительно пик, а не край спектра
            if (spectrum[i] > max(spectrum[i-1], spectrum[i+1]) or 
                spectrum[i] < min(spectrum[i-1], spectrum[i+1])):
                
                # Заменяем на линейную интерполяцию
                cleaned_spectrum[i] = 0.5 * (spectrum[i-1] + spectrum[i+1])
    
    return cleaned_spectrum

def normalize_by_area(df, x_values, cols=None, inplace=False):
    """
    Нормирует спектры по площади под кривой (интеграл = 1).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame со спектрами. Строки соответствуют образцам, столбцы — длинам волн.
    x_values : array-like
        Значения по оси X (длины волн, волновые числа и т.д.) для столбцов спектров.
        Должен иметь ту же длину, что и количество выбранных столбцов.
    cols : list or slice, optional
        Столбцы, содержащие спектральные данные. Если None, используются все столбцы.
    inplace : bool, default False
        Если True, изменяет исходный DataFrame. Иначе возвращает копию.

    Returns
    -------
    pd.DataFrame
        DataFrame с нормированными спектрами (интеграл по каждому спектру = 1).
        Неспектральные столбцы (если cols указан не полностью) остаются без изменений.
    """
    # Определяем столбцы для нормировки
    if cols is None:
        spec_cols = df.columns.tolist()
    else:
        spec_cols = df.columns[cols] if isinstance(cols, slice) else list(cols)
    
    # Преобразуем x_values в массив numpy и проверяем размерность
    x = np.asarray(x_values)
    if len(x) != len(spec_cols):
        raise ValueError(f"Длина x_values ({len(x)}) не совпадает с числом выбранных столбцов ({len(spec_cols)})")
    
    # Создаём копию, если нужно
    result = df if inplace else df.copy()
    
    # Для каждого спектра (строки) выполняем нормировку
    for idx in result.index:
        y = np.array(result.loc[idx, spec_cols].values.astype(float))
        
        # Вычисляем площадь методом трапеций
        area = np.trapezoid(y, x)
        
        # Защита от деления на ноль или очень малую площадь
        if area == 0 or np.isnan(area):
            # Можно оставить как есть или заполнить нулями — выбираем нули
            result.loc[idx, spec_cols] = 0.0
            # Предупреждение
            print(f"Предупреждение: площадь под спектром {idx} равна {area}. Значения обнулены.")
        else:
            # Нормируем: делим на площадь
            result.loc[idx, spec_cols] = y / area
    
    return result

# D-нормировка

def d_norm(data: Union[pd.Series, pd.DataFrame],
                laser_wave_left: float = 731.195121105382,
                laser_wave_right: float = 772.244519389919,
                cutoff_wave: float = 676.874469938485,
                mapping: bool = True):
    if isinstance(data, pd.Series):
        data = data.loc[:cutoff_wave].div(
            data.loc[laser_wave_left:laser_wave_right].mean())
    else:
        data = data.loc[:, :cutoff_wave].div(
            data.loc[:, laser_wave_left:laser_wave_right].mean(axis=1),
            axis=0)
        data.rename(axis=1,
                    mapper=lambda x: f"D{int(x)}" if mapping else None,
                    inplace=True)
    return data

# I-нормировка

def i_norm(data: pd.DataFrame, rename: bool=True) -> pd.DataFrame:
    data_Inorm_arr = []
    reg = LinearRegression()
    for group in data.index.get_level_values("GROUP").unique():
        data_buf = data[data.index.get_level_values("GROUP").isin([group])]
        mean_values = data_buf.mean().to_numpy().reshape(-1, 1)
        for i in range(len(data_buf)):
            x = np.array(data_buf.iloc[i]).reshape(-1, 1)
            reg.fit(x, mean_values)
            x = reg.predict(x)
            data_Inorm_arr.append(x.reshape(-1,))

    data_Inorm = np.asarray(data_Inorm_arr)
    data_Inorm = pd.DataFrame(data_Inorm, columns=data.keys(), index=data.index)
    if rename:
        data_Inorm.columns = [f"I{i[1:]}" if str(i)[0] == "D" else f"I{i}" for i in data_Inorm.keys()]
    return data_Inorm


# Чтение одного сырого файла спектрометра (оба шаблона: старый Parkinson-first-device
# и новый Optosky/dept1-2026) с автоопределением кодировки и разбором импульсов

def read_raw_spectrum_file(path: str) -> dict:
    """
    Read a single raw spectrometer .txt export, either template:
      - new (dept1 2026 / Sonya/dept2): UTF-8 with BOM, line 0 is a
        semicolon-delimited "key;value;key;value;..." metadata record,
        wavelengths line starts with "Pulses|Wavelenghts".
      - old (Parkinson first devise): UTF-16-LE, line 0 is "data/time:...",
        no structured metadata, wavelengths line starts with "Wavelenghts".

    Common layout after the metadata/equipment lines:
        line: n_points (int)
        line: n_pulses (int)
        line: gain/exposure parameter (unused here)
        line: "<Pulses|>Wavelenghts;wl_1;wl_2;...;wl_n_points"
        n_pulses lines: "<pulse_idx>;val_1;val_2;...;val_n_points"

    Returns
    -------
    dict with:
        metadata : dict[str, str]   (empty for the old template)
        equipment : str
        wavelengths : np.ndarray, shape (n_points,)
        spectra : np.ndarray, shape (n_pulses, n_points)
    """
    lines = None
    encoding_used = None
    for enc in ("utf-8-sig", "utf-16-le", "cp1251"):
        try:
            with open(path, "r", encoding=enc) as f:
                candidate = f.readlines()
            # a correctly-decoded file must have a numeric "n_points" on line 2
            if len(candidate) > 5 and candidate[2].strip().isdigit():
                lines = candidate
                encoding_used = enc
                break
        except (UnicodeDecodeError, UnicodeError):
            continue
    if lines is None:
        raise ValueError(f"could not decode {path!r} with any known encoding")

    line0 = lines[0].lstrip("﻿").rstrip("\n\r")
    equipment = lines[1].strip()

    metadata: dict[str, str] = {}
    if ";" in line0 and "ID исследования" in line0:
        parts = line0.split(";")
        metadata = dict(zip(parts[0::2], parts[1::2]))

    n_points = int(lines[2].strip())
    n_pulses = int(lines[3].strip())

    wl_line = lines[5].rstrip("\n\r")
    wl_tokens = wl_line.split(";")
    assert "Wavelenghts" in wl_tokens[0], f"unexpected wavelength header: {wl_tokens[0]!r}"
    wavelengths = np.array([float(t.replace(",", ".")) for t in wl_tokens[1:1 + n_points]])

    spectra = np.zeros((n_pulses, n_points))
    for i in range(n_pulses):
        row_tokens = lines[6 + i].rstrip("\n\r").split(";")
        spectra[i, :] = [float(t.replace(",", ".")) for t in row_tokens[1:1 + n_points]]

    return {
        "metadata": metadata,
        "equipment": equipment,
        "wavelengths": wavelengths,
        "spectra": spectra,
        "encoding": encoding_used,
    }


# Интерполяция одного или нескольких спектров на референсную сетку длин волн

def interpolate_spectra_to_grid(
        wavelengths: np.ndarray,
        spectra: np.ndarray,
        reference_wavelengths: np.ndarray,
        kind: str = "cubic",
        extrapolate: bool = False) -> np.ndarray:
    """
    Interpolate one or more spectra from `wavelengths` onto
    `reference_wavelengths`.

    Parameters
    ----------
    wavelengths : np.ndarray, shape (n_points,)
        Wavelengths of the source spectrum/spectra.
    spectra : np.ndarray, shape (n_points,) or (n_spectra, n_points)
        Intensities to interpolate.
    reference_wavelengths : np.ndarray, shape (n_ref,)
        Target wavelength grid.
    kind : str
        Passed to scipy.interpolate.interp1d (default: cubic spline).
    extrapolate : bool
        If False (default), reference points outside [wavelengths.min(),
        wavelengths.max()] are filled with NaN instead of extrapolated -
        cubic extrapolation far outside the observed range is unreliable
        and should not be silently fabricated as data.

    Returns
    -------
    np.ndarray, shape (n_ref,) or (n_spectra, n_ref)
    """
    spectra = np.asarray(spectra, dtype=float)
    single = spectra.ndim == 1
    if single:
        spectra = spectra[np.newaxis, :]

    fill_value = "extrapolate" if extrapolate else np.nan
    result = np.full((spectra.shape[0], len(reference_wavelengths)), np.nan)
    for i, y in enumerate(spectra):
        # interpolate row-by-row: interp1d's batched 2D-y mode propagates a
        # single NaN in one row to every row's output (verified empirically),
        # so a shared fit across all spectra at once is not safe here.
        valid = ~np.isnan(y)
        n_valid = int(valid.sum())
        min_points = {"cubic": 4, "quadratic": 3}.get(kind, 2)
        if n_valid < min_points:
            continue  # too few clean points for this interpolation order
        interp_func = interp1d(
            wavelengths[valid], y[valid], kind=kind,
            bounds_error=False, fill_value=fill_value)
        result[i] = interp_func(reference_wavelengths)

    return result[0] if single else result