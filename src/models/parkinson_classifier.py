import numpy as np
import pandas as pd
import pickle
import time
from sklearn.svm import SVC
from typing import Tuple, List, Dict, Optional, Union, Callable, Literal
from typing import Dict, Optional, Union
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_X_y, check_array
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin
from src.models.calibrator import *
from src.models.spectral_fcnn import SpectralClassifierFCNN
    


class ParkinsonClassifier(BaseEstimator, ClassifierMixin):
    """
    Классификатор для определения вероятности наличия болезни Паркинсона.
    Реализует ансамблевый подход с взвешенным объединением предсказаний базовых моделей.
    """
    
    def __init__(self, 
                 config: Dict,
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
        self.config = config
        models = config['models']
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
            'FCNN': SpectralClassifierFCNN,
        }
        self.available_calibs = {
            'SVC': SVCWithCalibration,
            'FCNN': NeuralNetCalibrator,
        }
        self.base_models = {}
        self.weights = {}
        self.calibs = {}

        # Инициализация моделей
        for model_name, model_params in models.items():
            model_name = model_name.upper()
            if model_name not in self.available_models:
                raise ValueError(f'{model_name} нет в списке доступных моделей. '
                               f'Доступные модели: {list(self.available_models.keys())}')
                
            try:
                self.base_models[model_name] = self.available_models[model_name](
                    **model_params.get('init_config', {})
                )
                self.weights[model_name] = model_params.get('weight', 1.0)
                if model_params['calibrator']:
                    if model_name in config['calibrators'].keys():
                        
                        self.base_models[model_name] = self.available_calibs[model_name](
                            self.base_models[model_name], 
                            model_config=model_params.get('init_config', {}), 
                            **config['calibrators'].get(['init_config'], {}),
                        )
                
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
            X_train: Union[pd.DataFrame, np.ndarray], 
            y_train: Union[pd.DataFrame, np.ndarray],
            calib_set: Tuple[Optional[Union[pd.DataFrame, np.ndarray]],
                             Optional[Union[pd.DataFrame, np.ndarray]]] = None,
            eval_set: Tuple[Optional[Union[pd.DataFrame, np.ndarray]],
                            Optional[Union[pd.DataFrame, np.ndarray]]] = None,
            config: Optional[Dict] = None,
        ) -> None:
        
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
        X_train, y_train = self._validate_data(X_train, y_train)
        if calib_set is not None:
            X_calib, y_calib = self._validate_data(calib_set[0], calib_set[1])
        else:
            X_calib, y_calib = (None, None)
        if eval_set is not None:
            X_test, y_test = self._validate_data(eval_set[0], eval_set[1])
        else:
            X_test, y_test = (None, None)
        self.classes_ = np.unique(y_train)
        self.X_train_ = X_train
        self.y_train_ = y_train
        self.X_calib_ = X_calib
        self.y_calib_ = y_calib
        self.X_test_ = X_test
        self.y_test_ = y_test
        
        # # Разделение данных если нужно
        # if self.train_config['test_size'] > 0:
        #     X_train, X_test, y_train, y_test = train_test_split(
        #         X, y,
        #         stratify=y,
        #         test_size=self.train_config['test_size'],
        #         random_state=self.train_config['random_state']
        #     )
        # else:
        #     X_train, y_train = X, y
        #     X_test, y_test = None, None
            
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
                if model in self.available_calibs:
                    model.fit(X_train, y_train, X_calib, y_calib)
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