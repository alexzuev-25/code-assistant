"""
API views для генерации и проверки кода
"""

from rest_framework.decorators import api_view
from rest_framework.response import Response

@api_view(['POST'])
def generate_code(request):
    """Генерация кода по текстовому описанию"""
    description = request.data.get('description', '')
    
    # Логика генерации кода
    if "сложение" in description.lower() and "python" in description.lower():
        generated_code = "def add(a, b):\n    return a + b"
    elif "факториал" in description.lower():
        generated_code = "def factorial(n):\n    if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)"
    else:
        generated_code = "# Функция будет сгенерирована здесь\n# Опишите задание более подробно"
    
    return Response({
        'status': 'success',
        'generated_code': generated_code,
        'description': description
    })

@api_view(['POST'])
def check_code(request):
    """Проверка предоставленного кода"""
    code = request.data.get('code', '')
    task_description = request.data.get('task_description', '')
    
    # Базовая проверка синтаксиса
    syntax_errors = []
    if "print(" in code and ")" not in code:
        syntax_errors.append("Незакрытая скобка в функции print")
    if "def " not in code and "функция" in task_description.lower():
        syntax_errors.append("Отсутствует определение функции")
    
    # Оценка качества кода
    quality_score = 85  # Пример оценки
    
    return Response({
        'status': 'success',
        'syntax_errors': syntax_errors,
        'quality_score': quality_score,
        'code_quality': 'good' if len(syntax_errors) == 0 else 'needs_improvement',
        'recommendations': [
            'Используйте осмысленные имена переменных',
            'Добавьте комментарии для сложных участков кода'
        ]
    })

@api_view(['POST'])
def login(request):
    """Аутентификация пользователя"""
    username = request.data.get('username', '')
    password = request.data.get('password', '')
    
    # Простая проверка учетных данных
    valid_users = {
        'teacher_test': {'password': 'Teacher123!', 'role': 'teacher'},
        'student_test': {'password': 'Student123!', 'role': 'student'},
        'admin_test': {'password': 'Admin123!', 'role': 'admin'}
    }
    
    if username in valid_users and password == valid_users[username]['password']:
        return Response({
            'status': 'success', 
            'role': valid_users[username]['role'],
            'message': f'Добро пожаловать, {username}!'
        })
    else:
        return Response({
            'status': 'error', 
            'message': 'Неверные учетные данные'
        }, status=401)