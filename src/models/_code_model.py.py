"""
Модель для работы с кодом и ML компонентами
"""

class CodeAssistantModel:
    """Класс для работы с моделью генерации кода"""
    
    def __init__(self):
        self.model_loaded = False
        self.model_name = "CodeGPT"
        self.version = "1.0"
    
    def load_model(self):
        """Загрузка предобученной модели"""
        try:
            # В реальном проекте здесь будет загрузка модели из файла
            # self.model = torch.load('model.pt')
            self.model_loaded = True
            return True
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            return False
    
    def generate_code(self, description, max_length=100):
        """Генерация кода по описанию"""
        if not self.model_loaded:
            self.load_model()
        
        # Заглушка для демонстрации
        templates = {
            'сложение': "def add(a, b):\n    return a + b",
            'факториал': "def factorial(n):\n    if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)",
            'сортировка': "def sort_list(lst):\n    return sorted(lst)"
        }
        
        for key, template in templates.items():
            if key in description.lower():
                return template
        
        return f"# Сгенерированный код для: {description}\n# Реализуйте функцию согласно описанию"
    
    def check_syntax(self, code):
        """Базовая проверка синтаксиса"""
        issues = []
        if "print(" in code and ")" not in code:
            issues.append("Синтаксическая ошибка: незакрытая скобка")
        if "def " in code and ":" not in code:
            issues.append("Синтаксическая ошибка: отсутствует двоеточие после определения функции")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'score': 100 - len(issues) * 10
        }