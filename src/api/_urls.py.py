"""
API маршруты для Code Assistant
"""

from django.urls import path
from . import views

urlpatterns = [
    path('generate/', views.generate_code, name='generate_code'),
    path('check/', views.check_code, name='check_code'),
    path('auth/login/', views.login, name='login'),
]