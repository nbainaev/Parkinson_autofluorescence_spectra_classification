import sys
import os
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