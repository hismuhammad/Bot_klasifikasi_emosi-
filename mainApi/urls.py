from django.urls import path, include
from . import views

urlpatterns = [
    path('ask-bot/', views.bot_response, name='ask_bot')
]