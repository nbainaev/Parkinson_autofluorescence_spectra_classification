import sys
import os
import scipy.stats as stats
import numpy as np
from pathlib import Path
from ruamel.yaml import YAML

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout  # Сохраняем оригинальный stdout
        if not os.path.exists('logs'):
            os.mkdir('logs')
        self.log = open(filename, "a", encoding="utf-8")  # Файл для логов

    def write(self, message):
        self.terminal.write(message)  # Вывод в консоль
        self.log.write(message)       # Запись в файл

    def flush(self):  # Метод для корректной работы с буферизацией
        self.terminal.flush()
        self.log.flush()

def read_config(filepath):
    if not isinstance(filepath, Path):
        filepath = Path(filepath)

    with filepath.open('r') as config_io:
        yaml = YAML()
        return yaml.load(config_io)

def save_yaml(data, path):
    """
    Сохраняет данные в YAML-файл по относительному пути.
    Требует, чтобы целевая папка уже существовала.
    
    Параметры:
        data (dict): Данные для сохранения
        path (str): Относительный путь (с расширением или без)
                   Пример: "configs/exp1" → сохранит в "configs/exp1.yaml"
    """
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.preserve_quotes = True
    yaml.allow_unicode = True


    if not path.endswith(('.yaml', '.yml')):
        path += '.yaml'

    
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f)

    print(f"Файл сохранён: {path}")

def compute_Z(data, start, stop):
    """
        Функция для рассчета Z-оценок по подгруппам Паркинсон, Контроль, Сравнение
        
        data: pd.DataFrame - данные
        start: pd.Index - начало исследуемого участка спектра
        stop: pd.Index - конец исследуемого участка
    """
    Z_cont_park = []
    Z_comp_park = []
    Z_comp_cont = []
    df_cont = data.loc[data.index.get_level_values(level='GROUP') == "Control", start:stop].copy()
    df_park = data.loc[data.index.get_level_values(level='GROUP') == "Parkinson", start:stop].copy()
    df_comp = data.loc[data.index.get_level_values(level='GROUP') == "Comparison", start:stop].copy()
    try:
        preds = [int(val)for val in df_cont.keys()]
    except:
        preds = [int(val[1:])for val in df_cont.keys()]
    
    n1 = len(df_cont)
    n2 = len(df_park)
    mU = n1 * n2 / 2
    sigma_U = (n1 * n2 * (n1 + n2 + 1) / 12) ** 0.5
    for predictor in df_cont.keys():
        U, p = stats.mannwhitneyu(df_cont[predictor], df_park[predictor])
        Z_cont_park.append((float(U) - mU) / sigma_U)
    n1 = len(df_comp)
    n2 = len(df_park)
    mU = n1 * n2 / 2
    sigma_U = (n1 * n2 * (n1 + n2 + 1) / 12) ** 0.5
    for predictor in df_cont.keys():
        U, p = stats.mannwhitneyu(df_comp[predictor], df_park[predictor])
        Z_comp_park.append((float(U) - mU) / sigma_U)
    n1 = len(df_comp)
    n2 = len(df_cont)
    mU = n1 * n2 / 2
    sigma_U = (n1 * n2 * (n1 + n2 + 1) / 12) ** 0.5
    for predictor in df_cont.keys():
        U, p = stats.mannwhitneyu(df_comp[predictor], df_cont[predictor])
        Z_comp_cont.append((float(U) - mU) / sigma_U)
    
    return Z_cont_park, Z_comp_park, Z_comp_cont, preds

def compute_t_Welch(data, start, stop, alpha=0.05):
    """
        Функция для рассчета t-оценок по подгруппам Паркинсон, Контроль, Сравнение
        
        data: pd.DataFrame - данные
        start: pd.Index - начало исследуемого участка спектра
        stop: pd.Index - конец исследуемого участка
    """
    t_cont_park = {"t_value": [], "p_value": [], "df": [], 'crit_t': []}
    t_comp_park = {"t_value": [], "p_value": [], "df": [], 'crit_t': []}
    t_comp_cont = {"t_value": [], "p_value": [], "df": [], 'crit_t': []}
    df_cont = data.loc[data.index.get_level_values(level='GROUP') == "Control", start:stop].copy()
    df_park = data.loc[data.index.get_level_values(level='GROUP') == "Parkinson", start:stop].copy()
    df_comp = data.loc[data.index.get_level_values(level='GROUP') == "Comparison", start:stop].copy()
    try:
        preds = [int(val)for val in df_cont.keys()]
    except:
        preds = [int(val[1:])for val in df_cont.keys()]
    
    for predictor in df_cont.keys():
        res = stats.ttest_ind(df_cont[predictor], df_park[predictor], equal_var=False, alternative="two-sided")
        t, p, df = res[0], res[1], res.df
        critical_value_two_sided = stats.t.ppf(1 - alpha/2, df)
        t_cont_park['t_value'].append(t)
        t_cont_park['p_value'].append(p)
        t_cont_park['df'].append(df)
        t_cont_park['crit_t'].append(critical_value_two_sided)

    for predictor in df_cont.keys():
        res = stats.ttest_ind(df_comp[predictor], df_park[predictor], equal_var=False, alternative="two-sided")
        t, p, df = res[0], res[1], res.df
        critical_value_two_sided = stats.t.ppf(1 - alpha/2, df)
        t_comp_park['t_value'].append(t)
        t_comp_park['p_value'].append(p)
        t_comp_park['df'].append(df)
        t_comp_park['crit_t'].append(critical_value_two_sided)

    for predictor in df_cont.keys():
        res = stats.ttest_ind(df_comp[predictor], df_cont[predictor], equal_var=False, alternative="two-sided")
        t, p, df = res[0], res[1], res.df
        critical_value_two_sided = stats.t.ppf(1 - alpha/2, df)
        t_comp_cont['t_value'].append(t)
        t_comp_cont['p_value'].append(p)
        t_comp_cont['df'].append(df)
        t_comp_cont['crit_t'].append(critical_value_two_sided)
    
    return {'cont_park': t_cont_park, 'comp_park': t_comp_park, 'comp_cont': t_comp_cont}, preds