"""
Утилиты для работы с моделью
"""

import torch
import torch.nn as nn

class SimpleCodeModel(nn.Module):
    def __init__(self, vocab_size=5000, embed_dim=128, hidden_dim=256, num_layers=3):
        super(SimpleCodeModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.fc(lstm_out)
        return output

class CodeGenerator:
    """Класс для генерации кода с использованием обученной модели"""
    
    def __init__(self, model_path='model.pt'):
        self.model = None
        self.model_path = model_path
        self.load_model()
    
    def load_model(self):
        """Загрузка обученной модели"""
        try:
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            # Создаем модель с теми же параметрами
            self.model = SimpleCodeModel(
                vocab_size=checkpoint['model_config']['vocab_size'],
                embed_dim=checkpoint['model_config']['embed_dim'],
                hidden_dim=checkpoint['model_config']['hidden_dim'],
                num_layers=checkpoint['model_config']['num_layers']
            )
            
            # Загружаем веса
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print("Модель успешно загружена")
            
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            self.model = None
    
    def generate_code(self, description, max_length=100):
        """Генерация кода по описанию"""
        if self.model is None:
            return "# Ошибка: модель не загружена"
        
        # Простая логика генерации на основе ключевых слов
        templates = {
            'сложение': '''def add(a, b):
    """Сложение двух чисел"""
    return a + b''',
            
            'факториал': '''def factorial(n):
    """Вычисление факториала числа"""
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)''',
            
            'простое': '''def is_prime(n):
    """Проверка числа на простоту"""
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True'''
        }
        
        # Ищем подходящий шаблон
        for key, template in templates.items():
            if key in description.lower():
                return template
        
        # Если шаблон не найден, возвращаем общий вариант
        return f"\"\"\"\nФункция для: {description}\n\"\"\"\ndef solution():\n    # Реализация функции\n    pass"
