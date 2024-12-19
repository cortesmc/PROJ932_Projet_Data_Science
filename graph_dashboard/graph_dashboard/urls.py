from django.urls import path
from graphs import views

urlpatterns = [
    path('dashboard/', views.graph_dashboard, name='graph_dashboard'),
    path('upload_file/', views.upload_file, name='upload_file'),
    ]
