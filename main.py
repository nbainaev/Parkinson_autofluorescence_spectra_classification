import sys
import os
import yaml
from src.model import *
import pandas as pd
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
# Перенаправляем stdout
sys.stdout = Logger("logs/output.log")

model_config_path = "configs/config.yaml"
config = dict()

with open(model_config_path, 'r') as file:
        config['models'] = yaml.load(file, Loader=yaml.Loader)

data = pd.read_excel('data/df_raw.xlsx', index_col=[0, 1, 2, 3])
X = data.groupby(level=['IDcard', 'GROUP']).mean()
y = pd.Series(np.array([1 if group == 'Parkinson' else 0 for group in X.index.get_level_values(level="GROUP")]))
model = ParkinsonClassifier(config['models'])
model.fit(X, y, {'verbose': True, 'metric': 'auc'})




# Возвращаем stdout обратно (если нужно)
sys.stdout = sys.stdout.terminal