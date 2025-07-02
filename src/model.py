import numpy as np
import random
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.svm import SVC
from typing import Tuple, List, Dict, Optional, Union, Callable, Literal
from typing import Dict, Optional, Union
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_X_y, check_array
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin
import pickle
import time

class Discriminator(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, activation: str="leaky_relu"):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim
        activations = {"leaky_rely": nn.LeakyReLU, "rely": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh}

        self.first_layer = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dims[0]),
            nn.BatchNorm1d(self.hidden_dims[0]),
            activations[activation](),
        )

        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
            nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]),
            nn.BatchNorm1d(self.hidden_dims[i + 1]),
            activations[activation](),
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

class SpectralClassifierFCNN(nn.Module):
    def __init__(self, 
                 in_dim=147, 
                 hidden_dims=[256, 128, 64], 
                 out_dim=1, 
                 activation="leaky_relu", 
                 dropout_rate=0.3, 
                 random_seed=None):
        super().__init__()
        
        activations = {
            "leaky_relu": nn.LeakyReLU(0.2),
            "relu": nn.ReLU(),
            "selu": nn.SELU(),
            "gelu": nn.GELU(),
            "sigmoid": nn.Sigmoid(),
            "silu": nn.SiLU()
        }
        if random_seed is not None:
            self.set_seed(random_seed) 
        
        self.first_block = nn.Sequential(
            nn.Linear(in_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.Dropout(dropout_rate),
            activations[activation]
        )
        
        self.hidden_blocks = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            block = nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.LayerNorm(hidden_dims[i+1]),
                nn.Dropout(dropout_rate),
                activations[activation]
            )
            self.hidden_blocks.append(block)
        
        self.output = nn.Sequential(
            nn.Linear(hidden_dims[-1], out_dim),
        )

        self._init_weights()

    def set_seed(self, seed=42):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def predict(self, val_dataloader):
        if isinstance(val_dataloader, DataLoader):
            pass
        elif isinstance(val_dataloader, TensorDataset):
            val_dataloader = DataLoader(val_dataloader, 
                                        batch_size=64, 
                                        shuffle=False)
                
        device = "cuda" \
            if next(self.parameters()).is_cuda \
            else "cpu"
        self.to(device)
        self.eval()
        preds_all = np.array([])
        proba_all = np.array([])
        labels_all = np.array([])
        for batch, labels in val_dataloader:
            with torch.no_grad():
                batch = batch.to(torch.float)
                logits = self(batch.to(device)).squeeze().to(torch.float)
                proba = torch.sigmoid(logits)
                preds = (proba > 0.5).to(torch.int)

                preds_all = np.concatenate((preds_all, 
                                            preds.detach().cpu().numpy()))
                labels_all = np.concatenate((labels_all, 
                                             labels.detach().cpu().numpy()))
                proba_all = np.concatenate((proba_all, 
                                            proba.detach().cpu().numpy()))

        return preds_all, proba_all, labels_all
    
    def create_default_dataloader(self, 
                                  X: pd.DataFrame, 
                                  y: pd.Series, 
                                  batch_size: int=32, 
                                  shuffle: bool=True, 
                                  worker_seed=42) -> DataLoader:

        labels = torch.tensor(y.to_numpy())
        data = torch.tensor(X.to_numpy(), dtype=torch.float)

        dataset = TensorDataset(data, labels)
        dataloader = DataLoader(dataset, 
                                batch_size=batch_size, 
                                shuffle=shuffle, 
                                worker_init_fn=worker_seed)
        return dataloader
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, 
                                        mode='fan_out', 
                                        nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.first_block(x)
        for block in self.hidden_blocks:
            residual = x
            x = block(x)
            if x.shape == residual.shape:
                x = x + residual
        return self.output(x)



class SVCWithCalibration:
    def __init__(self, config):
        self.base_model = SVC()


class ParkinsonClassifier:
    """
        Классификатор для определения вероятности наличия болезни Паркинсона.
    """
    def __init__(self, 
                 models: Dict,
                 train_config: Optional[Dict] = None
            ) -> None:
        """
            Инициализация модели
        
            Аргументы
            --------

            - models: Dict\n
                Список базовых моделей классификации. Список преставлен в виде словаря,\
                для каждой модели содержится словарь параметров инициализации.
            - train_config: Optional[Dict] = None\n
                Список параметров обучения для каждой модели.
        """
        if train_config is None:
            self.train_config = {} # Добавить стандартные значения
        else:
            self.train_config = train_config
        models = {'svc': SVCWithCalibration, 'fcnn': SpectralClassifierFCNN}
        self.base_models = {}
        self.weigths = {}

        # Инициализация моделей
        for model_name in models.keys():
            if model_name in models.values():
                self.base_models[model_name] = models[model_name](**models[model_name]['init_config'])
                self.weigths[model_name] = models[model_name]['weight']
                if 'train_config' in models[model_name].keys():
                    self.train_config[model_name] = models[model_name]['train_config']
            else:
                raise ValueError(f'{model_name} нет в списке доступных моделей. Доступные модели:{'{'}{', '.join(self.base_models.keys())}{'}'}')
        if not self.base_models:
            raise AssertionError(f'Не подана ни одна модель! Список доступных моделей: {'{'}{', '.join(self.base_models.keys())}{'}'}')
        
        self.weigths = np.array(self.weigths)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
            Функция предсказания наличия болезни Паркинсона.

            Аргументы:
            ---------

            - X: numpy.ndarray\n
                Входные данные (спектры аутофлуоресценции кожи пациентов). Имеет размер (n_samples, spectrum_length), где:
                - n_samples - количество входных спектров
                - spectrum_length - длина спектра
            
            Выходные данные:
            ---------------

            - preds: numpy.ndarray\n
                Предсказанные метки классов (0 - отсутствие болезни, 1 - наличие).\
                Имеет размерность (n_samples, )
        """
        pred_proba = []
        for i, model in enumerate(self.base_models):
            pred_proba.append(self.weigths[i] * model.predict_proba(X))
        
        pred_proba = sum(pred_proba)
        preds = (pred_proba > 0.5).astype(int)
        
        return preds
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
            Функция предсказания вероятности болезни Паркинсона.

            Аргументы:
            ---------

            - X: numpy.ndarray\n
                Входные данные (спектры аутофлуоресценции кожи пациентов). Имеет размер (n_samples, spectrum_length), где:
                - n_samples - количество входных спектров
                - spectrum_length - длина спектра
            
            Выходные данные:
            ---------------

            - pred_proba: numpy.ndarray\n
                Предсказанные вероятности принадлежности к положительному классу.\
                Имеет размерность (n_samples, ) 
        """
        pred_proba = []
        for i, model in enumerate(self.base_models):
            pred_proba.append(self.weigths[i] * model.predict_proba(X))
        
        pred_proba = sum(pred_proba)
        
        return pred_proba

    def fit(self, 
              X: Union[pd.DataFrame, np.ndarray], 
              y: Union[pd.DataFrame, np.ndarray],
              config: Optional[Dict]) -> None:
        """
            Функция для обучения базовых классификаторов:

            Аргументы:
            ---------

            - X: Union[pd.DataFrame, np.ndarray]\n
                Входные данные (спектры аутофлуоресценции кожи пациентов). Имеет размер (n_samples, spectrum_length), где:
                - n_samples - количество входных спектров
                - spectrum_length - длина спектра
            - y: Union[pd.DataFrame, np.ndarray]\n
                Истинные метки классов (0 - отсутствие болезни, 1 - наличие болезни)
            - train_config: Dict\n
                Список параметров обучения.
        """
        if self.train_config['test_size'] == 0:
            X_test, y_test = None, None
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                                stratify=y, 
                                                                test_size=self.train_config['test_size'],
                                                                random_state=self.random_state
                                                            )
        
        for model_name, model in self.base_models.items():
            model.train(X_train, y_train, X_test, y_test, **self.train_config[model_name])
    


class ParkinsonClassifier(BaseEstimator, ClassifierMixin):
    """
    Классификатор для определения вероятности наличия болезни Паркинсона.
    Реализует ансамблевый подход с взвешенным объединением предсказаний базовых моделей.
    """
    
    def __init__(self, 
                 models: Dict,
                 train_config: Optional[Dict] = None) -> None:
        """
        Инициализация модели
        
        Аргументы
        --------
        - models: Dict
            Словарь базовых моделей классификации. Для каждой модели должен содержаться:
            - 'init_config': параметры инициализации
            - 'weight': вес модели в ансамбле
            - (опционально) 'train_config': параметры обучения
        - train_config: Optional[Dict] = None
            Общие параметры обучения для всех моделей
        """
        # Установка конфига по умолчанию
        default_train_config = {
            'test_size': 0.2,
            'random_state': 42,
            'verbose': False,
            'metric': 'auc'
        }

        if train_config is None:
            self.train_config = default_train_config
        else:
            self.train_config = {**default_train_config, **train_config}
            
        # Доступные модели (можно расширять)
        self.available_models = {
            'SVC': SVC,
            'fcnn': SpectralClassifierFCNN,
        }
        
        self.base_models = {}
        self.weights = {}
        
        # Инициализация моделей
        for model_name, model_params in models.items():
            if model_name not in self.available_models:
                raise ValueError(f'{model_name} нет в списке доступных моделей. '
                               f'Доступные модели: {list(self.available_models.keys())}')
                
            try:
                self.base_models[model_name] = self.available_models[model_name](
                    **model_params.get('init_config', {})
                )
                self.weights[model_name] = model_params.get('weight', 1.0)
                
                # Сохраняем индивидуальные параметры обучения если они есть
                if 'train_config' in model_params:
                    self.train_config[model_name] = model_params['train_config']
                    
            except Exception as e:
                raise RuntimeError(f'Ошибка при инициализации модели {model_name}: {str(e)}')
        
        if not self.base_models:
            raise ValueError('Не подана ни одна модель!')
            
        # Нормализуем веса
        total_weight = sum(self.weights.values())
        self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
        self.metrics = {
            'accuracy': accuracy_score, 
            'precision': precision_score, 
            'recall': recall_score, 
            'f1': f1_score, 
            'auc': roc_auc_score
        }
        # Атрибуты для sklearn совместимости
        self.classes_ = None
        self.X_train_ = None
        self.y_train_ = None
        
    def _validate_data(self, X, y=None):
        """Валидация входных данных"""
        if y is None:
            return check_array(X, accept_sparse=True)
        return check_X_y(X, y, accept_sparse=True)
    
    def _check_fitted(self):
        """Проверка, что модель обучена"""
        if not hasattr(self, 'classes_'):
            raise RuntimeError('Модель не обучена! Сначала вызовите fit().')
    
    def fit(self, 
            X: Union[pd.DataFrame, np.ndarray], 
            y: Union[pd.DataFrame, np.ndarray],
            config: Optional[Dict] = None) -> None:
        """
        Обучение базовых классификаторов
        
        Аргументы:
        ---------
        - X: Входные данные (спектры аутофлуоресценции)
        - y: Метки классов (0 - здоров, 1 - болезнь Паркинсона)
        - config: Дополнительные параметры обучения
        """
        # Обновляем конфиг если передан
        if config is not None:
            self.train_config.update(config)
        # Валидация данных
        X, y = self._validate_data(X, y)
        self.classes_ = np.unique(y)
        self.X_train_ = X
        self.y_train_ = y
        
        # Разделение данных если нужно
        if self.train_config['test_size'] > 0:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                stratify=y,
                test_size=self.train_config['test_size'],
                random_state=self.train_config['random_state']
            )
        else:
            X_train, y_train = X, y
            X_test, y_test = None, None
            
        # Обучение моделей
        for model_name, model in self.base_models.items():
            if self.train_config['verbose']:
                print(f"Обучение {model_name}...")
                start_time = time.time()
                
            # Получаем индивидуальные параметры обучения если есть
            model_train_config = self.train_config.get(model_name, {})
            
            if hasattr(model, 'train'):
                model.train(X_train, y_train, X_test, y_test, **model_train_config)
            else:
                model.fit(X_train, y_train)
                
            if self.train_config['verbose']:
                print(f"{model_name} обучен за {time.time()-start_time:.2f} сек")

            if self.train_config['test_size'] > 0:
                print(f"Оценка {model_name} на тестовой выборке по метрике {self.train_config['metric']}:\
                    {self.metrics[self.train_config['metric']](y_test, model.predict(X_test))\
                        if self.train_config['metric'] != 'auc'\
                        else self.metrics[self.train_config['metric']](y_test, model.predict_proba(X_test)[:, 1]):.2f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказание наличия болезни Паркинсона
        
        Аргументы:
        ---------
        - X: Входные спектры (n_samples, spectrum_length)
        
        Возвращает:
        ----------
        - preds: Предсказанные метки классов (0 или 1)
        """
        return (self.predict_proba(X) >= 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказание вероятности болезни Паркинсона
        
        Аргументы:
        ---------
        - X: Входные спектры (n_samples, spectrum_length)
        
        Возвращает:
        ----------
        - pred_proba: Вероятности принадлежности к классу 1
        """
        self._check_fitted()
        X = self._validate_data(X)
        
        probas = []
        for model_name, model in self.base_models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)[:, 1]  # Берем вероятность класса 1
                else:
                    # Для моделей без predict_proba используем decision function
                    dec_func = model.decision_function(X)
                    proba = 1 / (1 + np.exp(-dec_func))  # Сигмоидная трансформация
                
                probas.append(self.weights[model_name] * proba)
                
            except Exception as e:
                raise RuntimeError(f"Ошибка при предсказании моделью {model_name}: {str(e)}")
        
        return np.sum(probas, axis=0)
    
    def score(self, X: np.ndarray, 
              y: np.ndarray, 
              metric: Union[Literal['accuracy', 'precision', 'recall', 'f1', 'roc-auc'], Callable]) -> float:
        """
        Оценка точности модели
        
        Аргументы:
        ---------
        - X: Входные данные
        - y: Истинные метки
        
        Возвращает:
        ----------
        - accuracy: Доля правильных ответов
        """
        if metric in self.metrics.keys():
            return self.metrics[metric](y, self.predict(X))
        elif callable(metric):
            return metric(y, self.predict(X))
        else:
            raise ValueError(f"Указанная метрика не является функцией или одной из поддерживаемых метрик ({list(self.metrics.keys())})")
    

    def get_params(self, deep=True) -> Dict:
        """Получение параметров модели (для совместимости с sklearn)"""
        return {
            'models': {name: {'weight': self.weights[name]} 
                      for name in self.base_models},
            'train_config': self.train_config
        }
    
    def set_params(self, **params) -> None:
        """Установка параметров модели"""
        if 'models' in params:
            for name, model_params in params['models'].items():
                if name in self.weights:
                    self.weights[name] = model_params.get('weight', self.weights[name])
        
        if 'train_config' in params:
            self.train_config.update(params['train_config'])
        
        return self
    
    def save(self, path: str) -> None:
        """Сохранение модели на диск"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str) -> 'ParkinsonClassifier':
        """Загрузка модели с диска"""
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def add_model(self, model_name: str, model, weight: float = 1.0) -> None:
        """
        Добавление новой модели в ансамбль
        
        Аргументы:
        ---------
        - model_name: Имя модели
        - model: Инициализированная модель
        - weight: Вес модели в ансамбле
        """
        if model_name in self.base_models:
            raise ValueError(f"Модель с именем {model_name} уже существует")
            
        self.base_models[model_name] = model
        self.weights[model_name] = weight
        # Перенормировка весов
        total_weight = sum(self.weights.values())
        self.weights = {k: v/total_weight for k, v in self.weights.items()}
    
    def remove_model(self, model_name: str) -> None:
        """Удаление модели из ансамбля"""
        if model_name not in self.base_models:
            raise ValueError(f"Модель {model_name} не найдена")
            
        del self.base_models[model_name]
        del self.weights[model_name]
        # Перенормировка весов
        if self.weights:
            total_weight = sum(self.weights.values())
            self.weights = {k: v/total_weight for k, v in self.weights.items()}