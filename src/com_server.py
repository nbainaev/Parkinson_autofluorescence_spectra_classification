import numpy as np
import pandas as pd
from src.models.parkinson_classifier import ParkinsonClassifier
from src.utils import Logger, read_config

class COMServer:
    _public_methods_ = ['Greet']
    _reg_clsid_ = "{F7BC4B6B-8421-42CF-9CEE-BB501CF44FEA}"  # Сгенерируйте свой GUID (можно использовать `python -c "import uuid; print(uuid.uuid1())"`)
    _reg_progid_ = "PythonCOM.Server"
    _reg_desc_ = "COM server on Python for Parkinson classification"
    
    def __init__(self, config_path=None, train_config_path=None):
        if config_path is not None:
            self.init_config = read_config(config_path)
        else:
            try:
                self.init_config = read_config("configs/models_config.yaml")
            except FileNotFoundError:
                raise FileNotFoundError("Файл не найден!")
        
        if train_config_path is not None:
            self.train_config = read_config(train_config_path)
        else:
            try:
                self.train_config = read_config("configs/train_config.yaml")
            except FileNotFoundError:
                raise FileNotFoundError("Файл не найден!")
        
        self.model = ParkinsonClassifier(self.init_config, self.train_config)

    def fit(self, X, y):
        pass
    
    def predict(self, X, y):
        pass
    
    def predict_proba(self, X, y):
        pass