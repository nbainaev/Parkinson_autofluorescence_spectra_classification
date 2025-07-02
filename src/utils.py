import sys
import os

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