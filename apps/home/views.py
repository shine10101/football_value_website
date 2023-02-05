# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django import template
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, HttpResponseRedirect
from django.template import loader
from django.urls import reverse
import pandas as pd


@login_required(login_url="/login/")
def index(request):
    context = {'segment': 'index'}
    html_template = loader.get_template('home/index.html')
    return HttpResponse(html_template.render(context, request))


@login_required(login_url="/login/")
def pages(request):
    context = {}
    # All resource paths end in .html.
    # Pick out the html file name from the url. And load that template.
    try:

        load_template = request.path.split('/')[-1]

        if load_template == 'admin':
            return HttpResponseRedirect(reverse('admin:index'))
        context['segment'] = load_template

        data = dataframe_view(request)
        context.update(data)

        html_template = loader.get_template('home/' + load_template)
        return HttpResponse(html_template.render(context, request))

    except template.TemplateDoesNotExist:

        html_template = loader.get_template('home/page-404.html')
        return HttpResponse(html_template.render(context, request))

    except:
        html_template = loader.get_template('home/page-500.html')
        return HttpResponse(html_template.render(context, request))

def dataframe_view(request):
    df = pd.read_csv(fr'C:\Users\Philip.Shine\PycharmProjects\valuebetting\predictions.csv')
    df = df[['Date', 'Time', 'HomeTeam', 'AwayTeam', 'Max_Value', 'Max_Value_Result']]
    df['Max_Value'] = (df['Max_Value']*100).round(1).astype(str)
    df['Max_Value'] = df['Max_Value'] + '%'
    data = df.to_dict(orient="records")
    return {"data": data}