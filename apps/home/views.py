# -*- encoding: utf-8 -*-

import datetime
import json
import logging
import os
import threading

from django import template
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from django.template import loader
from django.urls import reverse
from django.views.decorators.http import require_POST
import pandas as pd

logger = logging.getLogger(__name__)

LEAGUE_NAMES = {
    'E0': 'Premier League',
    'E1': 'Championship',
    'E2': 'League One',
    'E3': 'League Two',
    'EC': 'National League',
    'SC0': 'Scottish Premiership',
    'SC1': 'Scottish Championship',
    'SC2': 'Scottish League One',
    'SC3': 'Scottish League Two',
    'SP1': 'La Liga',
    'SP2': 'La Liga 2',
    'I1': 'Serie A',
    'I2': 'Serie B',
    'D1': 'Bundesliga',
    'D2': 'Bundesliga 2',
    'F1': 'Ligue 1',
    'F2': 'Ligue 2',
    'N1': 'Eredivisie',
    'B1': 'Jupiler League',
    'P1': 'Primeira Liga',
    'T1': 'Super Lig',
    'G1': 'Super League Greece',
}

# Module-level cache: avoids re-reading the CSV unless the file has changed.
_csv_cache = {'mtime': None, 'data': None, 'df': None}

# Track refresh state
_refresh_state = {'running': False, 'error': None, 'finished_at': None}


def _load_predictions():
    """Load predictions from CSV, using a mtime-based cache."""
    path = settings.PREDICTIONS_CSV
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        logger.error("Predictions CSV not found at %s", path)
        return _csv_cache['data'] or []

    if _csv_cache['mtime'] == mtime and _csv_cache['data'] is not None:
        return _csv_cache['data']

    try:
        df = pd.read_csv(path)
        columns = ['Div', 'Date', 'Time', 'HomeTeam', 'AwayTeam',
                    'HWin', 'Draw', 'AWin', 'Pred_FTR',
                    'Max_Value', 'Max_Value_Result', 'Pred_FTHG', 'Pred_FTAG']
        available = [c for c in columns if c in df.columns]
        df = df[available]

        # Add league names
        if 'Div' in df.columns:
            df['LeagueName'] = df['Div'].map(LEAGUE_NAMES).fillna(df['Div'])

        # Store raw DataFrame before formatting for aggregate computations
        _csv_cache['df'] = df.copy()

        # Format probabilities as percentages
        for col in ['HWin', 'Draw', 'AWin']:
            if col in df.columns:
                df[col] = (df[col] * 100).round(1).astype(str) + '%'

        # Keep raw Max_Value for sorting (already sorted in CSV), then format
        if 'Max_Value' in df.columns:
            df['Max_Value_Raw'] = df['Max_Value']
            df['Max_Value'] = (df['Max_Value'] * 100).round(1).astype(str) + '%'

        # Round predicted goals
        for col in ['Pred_FTHG', 'Pred_FTAG']:
            if col in df.columns:
                df[col] = df[col].round(1)

        data = df.to_dict(orient='records')
        _csv_cache['mtime'] = mtime
        _csv_cache['data'] = data
        logger.info("Loaded %d predictions from %s", len(data), path)
        return data
    except Exception:
        logger.exception("Failed to parse predictions CSV from %s", path)
        return _csv_cache['data'] or []


def _get_index_context():
    data = _load_predictions()
    raw_df = _csv_cache.get('df')

    last_updated = 'N/A'
    if _csv_cache['mtime']:
        last_updated = datetime.datetime.fromtimestamp(
            _csv_cache['mtime']
        ).strftime('%Y-%m-%d %H:%M')

    context = {
        'total_predictions': len(data),
        'last_updated': last_updated,
        'recent_predictions': data[:10],
        'top_picks': data[:5],
        'refresh_running': _refresh_state['running'],
    }

    if raw_df is not None and not raw_df.empty:
        positive_value_count = int((raw_df['Max_Value'] > 0).sum()) if 'Max_Value' in raw_df.columns else 0
        avg_value = round(raw_df['Max_Value'].mean() * 100, 1) if 'Max_Value' in raw_df.columns else 0

        # Top league by prediction count
        top_league = ''
        if 'Div' in raw_df.columns:
            top_league_code = raw_df['Div'].value_counts().index[0]
            top_league = LEAGUE_NAMES.get(top_league_code, top_league_code)

        # Prediction breakdown for doughnut chart
        pred_counts = {}
        if 'Pred_FTR' in raw_df.columns:
            pred_counts = raw_df['Pred_FTR'].value_counts().to_dict()

        # League breakdown for bar chart
        league_labels = []
        league_values = []
        if 'Div' in raw_df.columns:
            league_counts = raw_df['Div'].value_counts()
            league_labels = [LEAGUE_NAMES.get(k, k) for k in league_counts.index]
            league_values = league_counts.values.tolist()

        # Average value per league for bar chart
        avg_val_labels = []
        avg_val_values = []
        if 'Div' in raw_df.columns and 'Max_Value' in raw_df.columns:
            avg_value_per_league = raw_df.groupby('Div')['Max_Value'].mean().sort_values(ascending=False)
            avg_val_labels = [LEAGUE_NAMES.get(k, k) for k in avg_value_per_league.index]
            avg_val_values = [round(v * 100, 1) for v in avg_value_per_league.values]

        # Unique leagues for filter dropdown
        leagues = sorted(raw_df['Div'].unique().tolist()) if 'Div' in raw_df.columns else []
        league_options = [{'code': l, 'name': LEAGUE_NAMES.get(l, l)} for l in leagues]

        context.update({
            'positive_value_count': positive_value_count,
            'avg_value': avg_value,
            'top_league': top_league,
            'pred_counts_json': json.dumps(pred_counts),
            'league_labels_json': json.dumps(league_labels),
            'league_values_json': json.dumps(league_values),
            'avg_val_labels_json': json.dumps(avg_val_labels),
            'avg_val_values_json': json.dumps(avg_val_values),
            'league_options': league_options,
        })

    return context


def dataframe_view(request):
    data = _load_predictions()

    # League filter
    league_filter = request.GET.get('league', '')
    if league_filter:
        data = [d for d in data if d.get('Div') == league_filter]

    # Compute league options for the dropdown
    all_data = _load_predictions()
    leagues = sorted(set(d.get('Div', '') for d in all_data))
    league_options = [{'code': l, 'name': LEAGUE_NAMES.get(l, l)} for l in leagues]

    paginator = Paginator(data, 25)
    page_obj = paginator.get_page(request.GET.get('page', 1))
    return {
        'data': page_obj,
        'league_filter': league_filter,
        'league_options': league_options,
        'total_filtered': len(data),
    }


@login_required(login_url="/login/")
def index(request):
    context = {'segment': 'index'}
    context.update(_get_index_context())
    html_template = loader.get_template('home/index.html')
    return HttpResponse(html_template.render(context, request))


@login_required(login_url="/login/")
def pages(request):
    context = {}
    try:
        load_template = request.path.split('/')[-1]

        if load_template == 'admin':
            return HttpResponseRedirect(reverse('admin:index'))
        context['segment'] = load_template

        if load_template == 'tables.html':
            context.update(dataframe_view(request))

        html_template = loader.get_template('home/' + load_template)
        return HttpResponse(html_template.render(context, request))

    except template.TemplateDoesNotExist:
        html_template = loader.get_template('home/page-404.html')
        return HttpResponse(html_template.render(context, request))

    except Exception:
        logger.exception("Error rendering page %s", request.path)
        html_template = loader.get_template('home/page-500.html')
        return HttpResponse(html_template.render(context, request))


def _run_refresh():
    """Background thread target for running the prediction pipeline."""
    try:
        from apps.home.predictions.pipeline import run_predictions
        run_predictions()
        _refresh_state['error'] = None
    except Exception as e:
        logger.exception("Prediction refresh failed")
        _refresh_state['error'] = str(e)
    finally:
        _refresh_state['running'] = False
        _refresh_state['finished_at'] = datetime.datetime.now().isoformat()
        # Invalidate cache so next page load picks up new data
        _csv_cache['mtime'] = None


@require_POST
@login_required(login_url="/login/")
def refresh_predictions(request):
    """Start the prediction pipeline in a background thread."""
    if _refresh_state['running']:
        return JsonResponse({'status': 'already_running'})

    _refresh_state['running'] = True
    _refresh_state['error'] = None
    thread = threading.Thread(target=_run_refresh, daemon=True)
    thread.start()
    return JsonResponse({'status': 'started'})


@login_required(login_url="/login/")
def refresh_status(request):
    """Poll the current state of the refresh pipeline."""
    return JsonResponse({
        'running': _refresh_state['running'],
        'error': _refresh_state['error'],
        'finished_at': _refresh_state['finished_at'],
    })
