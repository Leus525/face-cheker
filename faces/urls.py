from django.urls import path, include
from . import views
from django.contrib.auth import views as auth_views
urlpatterns = [
  #  path(' ', views.as_view(), name='faces'),
    path('', views.show, name='faces'),
    path('new-face/', views.new, name='new'),
    path('compare-face/', views.compare, name='compare'),
]
