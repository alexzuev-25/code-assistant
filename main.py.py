"""
Основной файл приложения Code Assistant
Запускает веб-сервер и API для генерации и проверки кода
"""

import os
import sys
import django
from django.core.management import execute_from_command_line

# Добавляем путь к исходному коду
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Настройка Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'src.config.settings')
django.setup()

def start_server():
    """Запуск Django сервера"""
    print("Запуск Code Assistant...")
    print("Сервер доступен по адресу: http://localhost:8000")
    execute_from_command_line([__file__, 'runserver', '0.0.0.0:8000'])

if __name__ == "__main__":
    start_server()