import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, Optional, Union, List
from sklearn.metrics import roc_auc_score, roc_curve
from matplotlib import rcParams

def plot_intensity_histograms(data: pd.DataFrame, 
                             wave_list: list,
                             bins: int = 30,
                             figsize: tuple = (14, 8),
                             palette: str = "viridis",
                             kde: bool = True,
                             stat: str = "density",
                             common_norm: bool = False,
                             group_by: str = None) -> None:
    """
    Строит гистограммы распределения интенсивностей для заданных длин волн.
    
    Параметры:
    ----------
    data : pd.DataFrame
        DataFrame с спектральными данными (столбцы - длины волн, строки - измерения)
    wave_list : List[float]
        Список длин волн для анализа
    bins : int, optional
        Количество бинов в гистограмме (по умолчанию 30)
    figsize : tuple, optional
        Размер фигуры (по умолчанию (14, 8))
    palette : str, optional
        Цветовая палитра (по умолчанию "viridis")
    kde : bool, optional
        Если True, добавляет KDE оценку (по умолчанию True)
    stat : str, optional
        Тип статистики ("count", "frequency", "density", "probability")
    common_norm : bool, optional
        Нормализовать ли все гистограммы одинаково (по умолчанию False)
    group_by : str, optional
        Если указано, строит распределения для двух групп. Может быть именем колонки 
        или уровнем индекса. Если None, строит общее распределение (по умолчанию None)
    """
    
    # Проверка наличия длин волн в данных
    missing_waves = [w for w in wave_list if w not in data.columns]
    if missing_waves:
        raise ValueError(f"Длины волн {missing_waves} не найдены в данных")
    
    # Проверка наличия колонки/индекса для группировки
    if group_by is not None:
        if group_by not in data.columns and group_by not in data.index.names:
            raise ValueError(f"Колонка или индекс '{group_by}' не найдены в данных")
    
    # Настройка стиля
    sns.set_style("whitegrid")
    
    # Создаем сетку подграфиков
    n_waves = len(wave_list)
    n_cols = min(2, n_waves)
    n_rows = (n_waves + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_waves > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Цветовая палитра
    colors = sns.color_palette(palette, 2 if group_by else n_waves)
    
    # Находим общие пределы для одинакового масштабирования
    if common_norm:
        all_data = data[wave_list].values.flatten()
        x_min, x_max = np.nanmin(all_data), np.nanmax(all_data)
    
    for i, (wave, ax) in enumerate(zip(wave_list, axes)):
        if group_by is None:
            # Оригинальная логика - одно распределение
            wave_data = data[wave].dropna()
            
            sns.histplot(data=wave_data, 
                         bins=bins,
                         kde=kde,
                         color=colors[i],
                         stat=stat,
                         alpha=0.7,
                         edgecolor='white',
                         linewidth=0.5,
                         ax=ax)
            
            # Статистика для общего распределения
            stats_text = (f"μ = {wave_data.mean():.2f}\n"
                         f"σ = {wave_data.std():.2f}\n"
                         f"N = {len(wave_data)}")
        
        else:
            # Новая логика - два распределения по группам
            # Получаем данные для группировки
            if group_by in data.columns:
                group_data = data[[wave, group_by]].dropna()
                groups = group_data[group_by].unique()
            else:
                # Если group_by в индексе
                group_data = data[[wave]].copy()
                group_data[group_by] = data.index.get_level_values(group_by)
                group_data = group_data.dropna()
                groups = group_data[group_by].unique()
            
            if len(groups) != 2:
                raise ValueError(f"Ожидалось 2 группы, найдено {len(groups)}")
            
            # Строим распределения для двух групп
            for j, group in enumerate(groups):
                group_wave_data = group_data[group_data[group_by] == group][wave]
                
                sns.histplot(data=group_wave_data, 
                             bins=bins,
                             kde=kde,
                             color=colors[j],
                             stat=stat,
                             alpha=0.7,
                             edgecolor='white',
                             linewidth=0.5,
                             label=f'Group {group}',
                             ax=ax)
            
            # Добавляем легенду
            ax.legend(title=group_by)
            
            # Статистика для обеих групп
            stats_text = ""
            for j, group in enumerate(groups):
                group_wave_data = group_data[group_data[group_by] == group][wave]
                stats_text += (f"Gr.{group}: μ={group_wave_data.mean():.2f}, "
                              f"σ={group_wave_data.std():.2f}, "
                              f"N={len(group_wave_data)}\n")
        
        # Установка одинаковых пределов если нужно
        if common_norm:
            ax.set_xlim(x_min, x_max)
        
        # Настройка оформления
        # title_suffix = f" - groups по {group_by}" if group_by else ""
        ax.set_title(f'Wavelength: {wave[1:] if isinstance(wave, str) else int(wave)} nm', 
                    fontsize=12, pad=10)
        ax.set_xlabel('Intensity, a.u.', fontsize=12)
        ax.set_ylabel("Density, a.u.", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Добавляем аннотацию с базовой статистикой
        ax.annotate(stats_text, xy=(0.95, 0.95), 
                    xycoords='axes fraction',
                    ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=9 if group_by else 10)
    
    # Скрываем пустые subplots
    for j in range(n_waves, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_fluorescence_spectra(
    data: pd.DataFrame,
    x_col: str = 'wavelength',
    y_col: str = 'intensity',
    group_col: str = 'GROUP',
    groups: Optional[Dict[str, str]] = None,
    style: str = 'whitegrid',
    palette: Union[str, List[str], Dict[str, str]] = 'coolwarm',
    figsize: tuple = (12, 6),
    title: str = 'Средние спектры аутофлуоресценции кожи',
    xlabel: str = 'Длина волны (нм)',
    ylabel: str = 'Интенсивность (отн. ед.)',
    ci: int = 95,
    estimator: str = 'mean',
    translate: str = False,
    rotation: int = 45,
    grid_alpha: float = 0.3,
    legend_title: str = 'Группы',
    line_styles: Optional[Dict[str, tuple]] = None,  # Изменено на tuple
    line_width: int = 2,
    font_scale: float = 1.2,
    context: str = 'notebook',
    save_path: Optional[str] = None,
    **kwargs
):
    """
    Построение спектров аутофлуоресценции с настройкой стилей.
    
    Параметры:
    ----------
    line_styles : dict, optional
        Стили линий для групп в формате {'group_name': (on, off)}
        Например: {'Control': (2, 1), 'Diabetes': (4, 1)}
    """
    # Настройка стилей
    plt.style.use("seaborn-v0_8")
    sns.set_style(style)
    sns.set_context(context, font_scale=font_scale)
    
    if isinstance(palette, str) and groups:
        palette = list(groups.values())
    
    # Создание фигуры
    plt.figure(figsize=figsize)
    
    # Подготовка параметров стиля линий
    style_params = {}
    if line_styles:
        style_params['style'] = group_col
        style_params['dashes'] = line_styles
    
    # Построение графиков
    lineplot = sns.lineplot(
        data=data,
        x=x_col,
        y=y_col,
        hue=group_col,
        errorbar=('ci', ci),
        estimator=estimator,
        palette=palette,
        linewidth=line_width,
        **style_params,
        **kwargs
    )
    
    # Настройка оформления
    plt.title(title, pad=20, fontsize=20)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.xticks(rotation=rotation, fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(alpha=grid_alpha)
    
    # Настройка легенды
    handles, labels = lineplot.get_legend_handles_labels()
    if translate:
        translate = {"Control": "Контроль", "Parkinson": "Паркинсон", "Comparison": "Сравнение"}
        labels = [translate[val] for val in labels]
    plt.legend(
        title=legend_title,
        handles=handles,
        labels=labels,
        frameon=True,
        fontsize=14,
        framealpha=0.8
    )
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()

def draw_z(Z, Z_name, preds):
    # Настройка стиля
    plt.style.use('seaborn-v0_8')  # Используем актуальный стиль seaborn
    sns.set_theme(style="whitegrid")  # Альтернативный способ установки стиля
    
    # Создание фигуры с настроенным размером
    plt.figure(figsize=(12, 6), dpi=100)
    
    # Цветовая палитра (используем seaborn)
    colors = sns.color_palette("husl", len(Z))
    
    # Рисуем основные кривые
    for z, name, color in zip(Z, Z_name, colors):
        plt.plot(preds, z, 
                 linewidth=2.5,
                 alpha=0.9,
                 color=color,
                 label=name,
                 marker='o',
                 markersize=4,
                 markeredgecolor='white',
                 markeredgewidth=0.5)
    
    # Критическое значение Z
    plt.axhline(y=2.39, color="red", linestyle="--", linewidth=2)
    plt.axhline(y=-2.39, color="red", linestyle="--", linewidth=2)
    # Настройка осей и заголовков
    plt.xlabel("Wavelength, nm", fontsize=12, labelpad=10) 
    plt.ylabel("t score, a.u.", fontsize=12, labelpad=10)

    
    # Настройка сетки
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # Легенда
    legend = plt.legend(fontsize=10, framealpha=1, 
                       shadow=True, borderpad=1,
                       loc='upper right')
    legend.get_frame().set_facecolor('white')
    
    # Улучшенные пределы осей
    # plt.xlim(min(preds), max(preds))
    # y_max = max(np.max(Z), 2.39) * 1.1  # 10% запас сверху
    # plt.ylim(0, y_max)
    
    # Убираем верхнюю и правую рамку
    sns.despine()
    
    plt.tight_layout()
    plt.show()

def draw_t(t, t_name, preds):
    # Настройка стиля
    plt.style.use('seaborn-v0_8')  # Используем актуальный стиль seaborn
    sns.set_theme(style="whitegrid")  # Альтернативный способ установки стиля
    
    # Создание фигуры с настроенным размером
    plt.figure(figsize=(12, 6), dpi=100)
    
    # Цветовая палитра (используем seaborn)
    colors = sns.color_palette("husl", len(t))
    
    # Рисуем основные кривые
    for name, color in zip(t_name, colors):
        plt.plot(preds, t[name]['t_value'], 
                 linewidth=2.5,
                 alpha=0.9,
                 color=color,
                 label=name,
                 marker='o',
                 markersize=4,
                 markeredgecolor='white',
                 markeredgewidth=0.5)
    
    # Критическое значение Z
    plt.axhline(y=2.39, color="red", linestyle="--", linewidth=2)
    plt.axhline(y=-2.39, color="red", linestyle="--", linewidth=2)
    # Настройка осей и заголовков
    plt.title("D-нормировка", fontsize=16)
    plt.xlabel("Длина волны, нм", fontsize=12, labelpad=10) 
    plt.ylabel("Z оценка, усл. ед.", fontsize=12, labelpad=10)

    
    # Настройка сетки
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # Легенда
    legend = plt.legend(fontsize=10, framealpha=1, 
                       shadow=True, borderpad=1,
                       loc='upper right')
    legend.get_frame().set_facecolor('white')
    
    # Улучшенные пределы осей
    # plt.xlim(min(preds), max(preds))
    # y_max = max(np.max(Z), 2.39) * 1.1  # 10% запас сверху
    # plt.ylim(0, y_max)
    
    # Убираем верхнюю и правую рамку
    sns.despine()
    
    plt.tight_layout()
    plt.show()

def draw_roc_curve(y_true: np.ndarray, 
                   proba_all: list, 
                   y_pred_all: list, 
                   names: list,
                   figsize=(10, 8),
                   linewidth=2,
                   diagonal_alpha=0.5,
                   style='ggplot',
                   palette='viridis'):
    """
    Улучшенная визуализация ROC-кривых для нескольких моделей
    
    Параметры:
    ----------
    y_true : np.ndarray
        Истинные метки классов (1D массив)
    proba_all : list
        Список предсказанных вероятностей для положительного класса
    y_pred_all : list
        Список предсказанных классов
    names : list
        Названия моделей (для легенды)
    figsize : tuple
        Размер графика (ширина, высота)
    linewidth : int
        Толщина линий ROC-кривых
    diagonal_alpha : float
        Прозрачность диагональной пунктирной линии
    style : str
        Стиль matplotlib (доступные: 'ggplot', 'seaborn', 'default' и др.)
    palette : str
        Название цветовой палитры ('viridis', 'plasma', 'tab10' и др.)
    """
    
    # Проверка доступных стилей
    available_styles = plt.style.available
    if style not in available_styles:
        print(f"Стиль '{style}' недоступен. Используются стандартные настройки.")
        print(f"Доступные стили: {available_styles}")
        style = 'default'
    
    # Применение стиля
    plt.style.use(style)
    
    # Настройка параметров
    rcParams.update({
        'axes.titlesize': 20,
        'axes.labelsize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 14,
        'grid.alpha': 0.3
    })
    
    # Создание фигуры
    fig, ax = plt.subplots(figsize=figsize)
    
    # Диагональная линия (random classifier)
    ax.plot([0, 1], [0, 1], 
            linestyle='--', 
            color='gray', 
            alpha=diagonal_alpha,
            label='Случайный классификатор (AUC = 0.5)')
    
    # Получение цветов из палитры
    # colors = plt.get_cmap(palette, len(names))
    colors = [
    "#E6194B",  # Красный
    "#3CB44B",  # Зелёный
    "#4363D8",  # Синий
    "#FFE119",  # Жёлтый
    "#F032E6",  # Пурпурный
    "#42D4F4"   # Голубой
]
    
    # Построение ROC-кривых для каждой модели
    for i, (proba, y_pred, name) in enumerate(zip(proba_all, y_pred_all, names)):
        auc_score = roc_auc_score(y_true, proba)
        fpr, tpr, _ = roc_curve(y_true, proba, pos_label=1)
        
        ax.plot(fpr, tpr, 
                linewidth=linewidth,
                color=colors[i],
                label=f'{name} (AUC = {auc_score:.3f})')
    
    # Настройка отображения
    ax.set_title('ROC-кривые моделей', pad=20)
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.grid(True)
    ax.legend(frameon=True, framealpha=0.9, fontsize=16)
    
    # Добавление информации о данных
    # ax.annotate(f'Общее количество: {len(y_true)}\nПоложительный класс: {sum(y_true)}\nОтрицательный класс: {len(y_true)-sum(y_true)}',
    #             xy=(0.98, 0.02), 
    #             xycoords='axes fraction',
    #             ha='right', 
    #             va='bottom',
    #             fontsize=12,
    #             bbox=dict(boxstyle='round', alpha=0.2))
    
    # Установка границ
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    
    # Улучшенное отображение
    plt.tight_layout()
    plt.show()

