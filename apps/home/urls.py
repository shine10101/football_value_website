# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django.urls import path, re_path
from apps.home import views

urlpatterns = [

    # The home page
    path('', views.index, name='home'),

    # Performance sub-pages — Match Result
    path('performance/', views.performance, name='performance'),
    path('performance/accuracy/', views.performance_accuracy, name='performance_accuracy'),
    path('performance/financial/', views.performance_financial, name='performance_financial'),
    path('performance/strategies/', views.performance_strategies, name='performance_strategies'),

    # Performance sub-pages — BTTS
    path('performance/btts/', views.performance_btts_accuracy, name='performance_btts_accuracy'),

    # Performance sub-pages — Over/Under 2.5
    path('performance/overunder/', views.performance_ou_accuracy, name='performance_ou_accuracy'),
    path('performance/overunder/financial/', views.performance_ou_financial, name='performance_ou_financial'),
    path('performance/overunder/strategies/', views.performance_ou_strategies, name='performance_ou_strategies'),

    # Methodology
    path('methodology/', views.methodology, name='methodology'),

    # Prediction refresh API
    path('api/refresh/', views.refresh_predictions, name='refresh_predictions'),
    path('api/refresh/status/', views.refresh_status, name='refresh_status'),

    # Matches any html file
    re_path(r'^.*\.*', views.pages, name='pages'),

]
