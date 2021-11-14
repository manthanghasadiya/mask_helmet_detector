from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('about', views.about, name='about'),
    path('helmetDetection', views.helmetDetection, name='helmetDetection'),
    path('maskDetection', views.maskDetection, name='maskDetection'),
]
