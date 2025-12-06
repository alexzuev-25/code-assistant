"""
Основные URL маршруты приложения
"""

from django.urls import path, include

urlpatterns = [
    path('api/', include('src.api.urls')),
]