from django.http.response import HttpResponse
from django.shortcuts import render
from /rain-models. import helmet


# Create your views here.


def home(request):
    return render(request, 'base.html')


def about(request):
    return render(request, 'about.html')

def maskDetection(request):
    return render(request, 'maskdetection.html')

def helmetDetection(request):
    
    return render(request,'helmetDetection.html')