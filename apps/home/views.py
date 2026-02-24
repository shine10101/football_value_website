# -*- encoding: utf-8 -*-

import datetime
import json
import logging
import os
import threading
from collections import defaultdict

from django import template
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.db.models import F, Q
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


def _get_team_accuracy_map():
    """Return accuracy maps for teams, leagues, and value bet win rates.

    Returns a dict with:
      - team names -> {correct, total, accuracy}
      - '__league__<div>' -> {correct, total, accuracy}
      - '__vb__<team>' -> {wins, total, winrate}
    """
    from apps.home.models import Prediction
    from apps.home.strategies import RESULT_MAP

    stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    for p in Prediction.objects.filter(resolved=True).only(
        'home_team', 'away_team', 'pred_ftr', 'actual_ftr',
        'div', 'max_value', 'max_value_result',
    ):
        is_correct = (p.pred_ftr == p.actual_ftr)

        # Team prediction accuracy
        for team in [p.home_team, p.away_team]:
            stats[team]['total'] += 1
            if is_correct:
                stats[team]['correct'] += 1

        # League prediction accuracy
        league_key = '__league__' + p.div
        stats[league_key]['total'] += 1
        if is_correct:
            stats[league_key]['correct'] += 1

        # Per-team value bet win rate
        if p.max_value and p.max_value > 0:
            vb_outcome = RESULT_MAP.get(p.max_value_result)
            vb_won = (vb_outcome == p.actual_ftr)
            for team in [p.home_team, p.away_team]:
                vb_key = '__vb__' + team
                stats[vb_key]['total'] += 1
                if vb_won:
                    stats[vb_key]['wins'] = stats[vb_key].get('wins', 0) + 1

    # Compute accuracy/winrate percentages
    for key, s in stats.items():
        if key.startswith('__vb__'):
            s.setdefault('wins', 0)
            s['winrate'] = round(s['wins'] / s['total'] * 100, 1) if s['total'] > 0 else 0
        else:
            s['accuracy'] = round(s['correct'] / s['total'] * 100, 1) if s['total'] > 0 else 0
    return dict(stats)


def _compute_confidence_flags(raw_df, accuracy_map):
    """Check each CSV row against the full confidence pick criteria.

    Applies all 6 filters: value, model-market agreement, decisive probability,
    team predictability, league predictability, odds range, value bet win rate.

    Returns a list of booleans, one per row.
    """
    result_to_prob_col = {'Home Win': 'HWin', 'Draw': 'Draw', 'Away Win': 'AWin'}
    result_to_pred = {'Home Win': 'H', 'Draw': 'D', 'Away Win': 'A'}

    flags = []
    for _, row in raw_df.iterrows():
        # 0. Value > 0
        has_value = (row.get('Max_Value', 0) or 0) > 0
        if not has_value:
            flags.append(False)
            continue

        bet_result = row.get('Max_Value_Result', '')
        prob_col = result_to_prob_col.get(bet_result)
        if not prob_col or prob_col not in raw_df.columns:
            flags.append(False)
            continue

        bet_prob = row.get(prob_col, 0) or 0

        # 1. Model-market agreement: value pick matches most-likely outcome
        h_prob = row.get('HWin', 0) or 0
        d_prob = row.get('Draw', 0) or 0
        a_prob = row.get('AWin', 0) or 0
        probs = {'H': h_prob, 'D': d_prob, 'A': a_prob}
        max_prob_outcome = max(probs, key=probs.get)
        bet_code = result_to_pred.get(bet_result)
        if max_prob_outcome != bet_code:
            flags.append(False)
            continue

        # 2. Decisive probability: > 50% and >= 10pp margin over next best
        sorted_probs = sorted(probs.values(), reverse=True)
        margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) >= 2 else 0
        if bet_prob <= 0.50 or margin < 0.10:
            flags.append(False)
            continue

        # 3. Team predictability: both >= 60% accuracy, min 5 matches
        home = row.get('HomeTeam', '')
        away = row.get('AwayTeam', '')
        home_stats = accuracy_map.get(home, {'total': 0, 'accuracy': 0})
        away_stats = accuracy_map.get(away, {'total': 0, 'accuracy': 0})
        if not (home_stats['total'] >= 5 and home_stats.get('accuracy', 0) >= 50.0):
            flags.append(False)
            continue
        if not (away_stats['total'] >= 5 and away_stats.get('accuracy', 0) >= 50.0):
            flags.append(False)
            continue

        # 4. League predictability: >= 50% accuracy, min 20 matches
        div = row.get('Div', '')
        league_stats = accuracy_map.get('__league__' + div, {'total': 0, 'accuracy': 0})
        if not (league_stats['total'] >= 20 and league_stats.get('accuracy', 0) >= 40.0):
            flags.append(False)
            continue

        # 5. Odds range: 1.4 - 4.0 (avoid extreme longshots and tiny payouts)
        best_odds = row.get('Best_Odds', None)
        if best_odds is None or not (1.4 <= best_odds <= 4.0):
            flags.append(False)
            continue

        # 6. Value bet win rate: skip teams with poor value bet record (< 40%, min 3 bets)
        home_vb = accuracy_map.get('__vb__' + home, {'total': 0, 'wins': 0, 'winrate': 0})
        away_vb = accuracy_map.get('__vb__' + away, {'total': 0, 'wins': 0, 'winrate': 0})
        if home_vb['total'] >= 3 and home_vb.get('winrate', 0) < 40.0:
            flags.append(False)
            continue
        if away_vb['total'] >= 3 and away_vb.get('winrate', 0) < 40.0:
            flags.append(False)
            continue

        flags.append(True)
    return flags


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
                    'BTTS_Yes', 'BTTS_No', 'Over25', 'Under25',
                    'Over25_Value', 'Under25_Value',
                    'OU_Max_Value', 'OU_Best_Bet', 'OU_Best_Odds',
                    'Best_Odds', 'Odds_Over25', 'Odds_Under25',
                    'Max_Value', 'Max_Value_Result', 'Pred_FTHG', 'Pred_FTAG']
        available = [c for c in columns if c in df.columns]
        df = df[available]

        # Add league names
        if 'Div' in df.columns:
            df['LeagueName'] = df['Div'].map(LEAGUE_NAMES).fillna(df['Div'])

        # Store raw DataFrame before formatting for aggregate computations
        _csv_cache['df'] = df.copy()

        # Keep raw BTTS_Yes for sorting
        if 'BTTS_Yes' in df.columns:
            df['BTTS_Yes_Raw'] = df['BTTS_Yes']

        # Format probabilities as percentages
        for col in ['HWin', 'Draw', 'AWin', 'BTTS_Yes', 'BTTS_No', 'Over25', 'Under25']:
            if col in df.columns:
                df[col] = (df[col] * 100).round(1).astype(str) + '%'

        # Format O/U value columns as percentages
        for col in ['Over25_Value', 'Under25_Value']:
            if col in df.columns:
                df[col + '_Raw'] = df[col]
                df[col] = (df[col] * 100).round(1).astype(str) + '%'

        # Keep raw OU_Max_Value for sorting, then format
        if 'OU_Max_Value' in df.columns:
            df['OU_Max_Value_Raw'] = df['OU_Max_Value']
            df['OU_Max_Value'] = (df['OU_Max_Value'] * 100).round(1).astype(str) + '%'

        # Round odds to 2 decimal places
        for col in ['Best_Odds', 'Odds_Over25', 'Odds_Under25', 'OU_Best_Odds']:
            if col in df.columns:
                df[col] = df[col].round(2)

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

    # Compute confidence picks
    confidence_picks = []
    confidence_pick_count = 0
    if raw_df is not None and not raw_df.empty:
        team_accuracy = _get_team_accuracy_map()
        confidence_flags = _compute_confidence_flags(raw_df, team_accuracy)
        for i, row in enumerate(data):
            if i < len(confidence_flags):
                row['is_confidence_pick'] = confidence_flags[i]
                if confidence_flags[i]:
                    confidence_pick_count += 1
                    if len(confidence_picks) < 5:
                        confidence_picks.append(row)

    context = {
        'total_predictions': len(data),
        'last_updated': last_updated,
        'recent_predictions': data[:10],
        'top_picks': data[:5],
        'confidence_picks': confidence_picks,
        'confidence_pick_count': confidence_pick_count,
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

        # O/U stats
        ou_value_count = int((raw_df['OU_Max_Value'] > 0).sum()) if 'OU_Max_Value' in raw_df.columns else 0
        ou_avg_value = round(raw_df['OU_Max_Value'].mean() * 100, 1) if 'OU_Max_Value' in raw_df.columns else 0
        ou_over_count = 0
        ou_under_count = 0
        if 'OU_Best_Bet' in raw_df.columns:
            ou_over_count = int((raw_df['OU_Best_Bet'] == 'Over 2.5').sum())
            ou_under_count = int((raw_df['OU_Best_Bet'] == 'Under 2.5').sum())

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
            'ou_value_count': ou_value_count,
            'ou_avg_value': ou_avg_value,
            'ou_over_count': ou_over_count,
            'ou_under_count': ou_under_count,
        })

    return context


def dataframe_view(request):
    data = _load_predictions()
    raw_df = _csv_cache.get('df')

    # Annotate confidence pick flags
    if raw_df is not None and not raw_df.empty:
        team_accuracy = _get_team_accuracy_map()
        confidence_flags = _compute_confidence_flags(raw_df, team_accuracy)
        for i, row in enumerate(data):
            if i < len(confidence_flags):
                row['is_confidence_pick'] = confidence_flags[i]

    # League filter
    league_filter = request.GET.get('league', '')
    if league_filter:
        data = [d for d in data if d.get('Div') == league_filter]

    # Market filter: 'result' (default), 'btts', 'overunder'
    market = request.GET.get('market', 'result')
    if market not in ('result', 'btts', 'overunder'):
        market = 'result'

    # Sort by the relevant value column for the selected market
    if market == 'overunder':
        data = sorted(data, key=lambda d: d.get('OU_Max_Value_Raw') or -999, reverse=True)
    elif market == 'btts':
        # Sort by BTTS probability (no value calc yet)
        data = sorted(data, key=lambda d: d.get('BTTS_Yes_Raw', 0) or 0, reverse=True)
    # else: default CSV sort by Max_Value is already correct

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
        'market': market,
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


def _get_base_performance_qs(request):
    """Shared: parse league filter, return (queryset, league_filter, league_options)."""
    from apps.home.models import Prediction

    league_filter = request.GET.get('league', '')
    qs = Prediction.objects.filter(resolved=True)
    if league_filter:
        qs = qs.filter(div=league_filter)

    all_leagues = Prediction.objects.filter(resolved=True).values_list('div', flat=True).distinct()
    league_options = sorted(
        [{'code': l, 'name': LEAGUE_NAMES.get(l, l)} for l in all_leagues],
        key=lambda x: x['name'],
    )
    return qs, league_filter, league_options


def _get_accuracy_context(request):
    """Build context for the model accuracy page."""
    qs, league_filter, league_options = _get_base_performance_qs(request)

    # Additional filters specific to accuracy results table
    team_filter = request.GET.get('team', '').strip()
    result_filter = request.GET.get('result', '')

    filtered_qs = qs
    if team_filter:
        filtered_qs = filtered_qs.filter(Q(home_team__icontains=team_filter) | Q(away_team__icontains=team_filter))
    if result_filter == 'correct':
        filtered_qs = filtered_qs.filter(pred_ftr=F('actual_ftr'))
    elif result_filter == 'incorrect':
        filtered_qs = filtered_qs.exclude(pred_ftr=F('actual_ftr'))

    total_resolved = filtered_qs.count()

    # Overall accuracy
    correct_count = filtered_qs.filter(pred_ftr=F('actual_ftr')).count()
    overall_accuracy = round(correct_count / total_resolved * 100, 1) if total_resolved > 0 else 0

    # Value bet accuracy (max_value > 0)
    value_qs = filtered_qs.filter(max_value__gt=0)
    value_total = value_qs.count()
    value_correct = value_qs.filter(pred_ftr=F('actual_ftr')).count()
    value_accuracy = round(value_correct / value_total * 100, 1) if value_total > 0 else 0

    # Value bet win rate (did the max_value_result outcome actually happen?)
    value_bet_wins = 0
    for p in value_qs.only('max_value_result', 'actual_ftr'):
        result_map = {'Home Win': 'H', 'Draw': 'D', 'Away Win': 'A'}
        if result_map.get(p.max_value_result) == p.actual_ftr:
            value_bet_wins += 1
    value_bet_win_rate = round(value_bet_wins / value_total * 100, 1) if value_total > 0 else 0

    # Accuracy by league
    league_stats = {}
    for div in filtered_qs.values_list('div', flat=True).distinct():
        league_qs = filtered_qs.filter(div=div)
        league_total = league_qs.count()
        league_correct = league_qs.filter(pred_ftr=F('actual_ftr')).count()
        league_stats[div] = {
            'name': LEAGUE_NAMES.get(div, div),
            'total': league_total,
            'correct': league_correct,
            'accuracy': round(league_correct / league_total * 100, 1) if league_total > 0 else 0,
        }
    league_stats_sorted = sorted(league_stats.values(), key=lambda x: x['accuracy'], reverse=True)

    # Calibration data: bucket predicted probability of the predicted outcome
    calibration_buckets = defaultdict(lambda: {'count': 0, 'correct': 0})
    for p in filtered_qs.only('h_win', 'draw', 'a_win', 'pred_ftr', 'actual_ftr'):
        prob_map = {'H': p.h_win, 'D': p.draw, 'A': p.a_win}
        pred_prob = prob_map.get(p.pred_ftr, 0)
        bucket = min(int(pred_prob * 10), 9)
        calibration_buckets[bucket]['count'] += 1
        if p.pred_ftr == p.actual_ftr:
            calibration_buckets[bucket]['correct'] += 1

    cal_labels = []
    cal_predicted = []
    cal_actual = []
    for i in range(10):
        cal_labels.append(f"{i*10}-{(i+1)*10}%")
        mid = (i * 10 + (i + 1) * 10) / 2
        cal_predicted.append(mid)
        bucket_data = calibration_buckets[i]
        if bucket_data['count'] > 0:
            cal_actual.append(round(bucket_data['correct'] / bucket_data['count'] * 100, 1))
        else:
            cal_actual.append(None)

    # Value bucket analysis
    value_buckets_def = [
        ('<0%', -999, 0),
        ('0-5%', 0, 0.05),
        ('5-10%', 0.05, 0.10),
        ('10-15%', 0.10, 0.15),
        ('15%+', 0.15, 999),
    ]
    value_bucket_labels = []
    value_bucket_accuracy = []
    value_bucket_counts = []
    for label, low, high in value_buckets_def:
        bucket_qs = filtered_qs.filter(max_value__gte=low, max_value__lt=high)
        count = bucket_qs.count()
        correct = bucket_qs.filter(pred_ftr=F('actual_ftr')).count()
        value_bucket_labels.append(label)
        value_bucket_accuracy.append(round(correct / count * 100, 1) if count > 0 else 0)
        value_bucket_counts.append(count)

    # Accuracy over time (by month)
    time_labels = []
    time_accuracy = []
    months = filtered_qs.dates('date', 'month', order='ASC')
    for month in months:
        month_qs = filtered_qs.filter(date__year=month.year, date__month=month.month)
        month_total = month_qs.count()
        month_correct = month_qs.filter(pred_ftr=F('actual_ftr')).count()
        time_labels.append(month.strftime('%b %Y'))
        time_accuracy.append(round(month_correct / month_total * 100, 1) if month_total > 0 else 0)

    # Team performance
    team_stats = defaultdict(lambda: {'matches': 0, 'correct': 0})
    for p in filtered_qs.only('home_team', 'away_team', 'pred_ftr', 'actual_ftr', 'max_value'):
        is_correct = 1 if p.pred_ftr == p.actual_ftr else 0
        team_stats[p.home_team]['matches'] += 1
        team_stats[p.home_team]['correct'] += is_correct
        team_stats[p.away_team]['matches'] += 1
        team_stats[p.away_team]['correct'] += is_correct

    team_stats_list = []
    for team, stats in sorted(team_stats.items()):
        stats['team'] = team
        stats['accuracy'] = round(stats['correct'] / stats['matches'] * 100, 1) if stats['matches'] > 0 else 0
        team_stats_list.append(stats)
    team_stats_list.sort(key=lambda x: x['accuracy'], reverse=True)

    # Paginated results table
    results_qs = filtered_qs.order_by('-date', 'div')
    paginator = Paginator(results_qs, 25)
    page_obj = paginator.get_page(request.GET.get('page', 1))

    return {
        'total_resolved': total_resolved,
        'overall_accuracy': overall_accuracy,
        'correct_count': correct_count,
        'value_total': value_total,
        'value_accuracy': value_accuracy,
        'value_bet_win_rate': value_bet_win_rate,
        'league_stats': league_stats_sorted,
        'cal_labels_json': json.dumps(cal_labels),
        'cal_predicted_json': json.dumps(cal_predicted),
        'cal_actual_json': json.dumps(cal_actual),
        'value_bucket_labels_json': json.dumps(value_bucket_labels),
        'value_bucket_accuracy_json': json.dumps(value_bucket_accuracy),
        'value_bucket_counts_json': json.dumps(value_bucket_counts),
        'time_labels_json': json.dumps(time_labels),
        'time_accuracy_json': json.dumps(time_accuracy),
        'team_stats': team_stats_list,
        'results_page': page_obj,
        'league_filter': league_filter,
        'team_filter': team_filter,
        'result_filter': result_filter,
        'league_options': league_options,
    }


def _get_financial_context(request):
    """Build context for the financial analysis page."""
    qs, league_filter, league_options = _get_base_performance_qs(request)

    strategy_filter = request.GET.get('strategy', '')

    from apps.home.strategies import simulate_all, STRATEGIES
    strategy_options = [{'id': s['id'], 'name': s['name']} for s in STRATEGIES]

    # Fetch resolved predictions ordered by date
    financial_qs = qs.order_by('date')
    financial_preds = list(financial_qs.only(
        'date', 'div', 'home_team', 'away_team',
        'h_win', 'draw', 'a_win',
        'odds_h', 'odds_d', 'odds_a',
        'max_value', 'max_value_result', 'actual_ftr',
    ))

    # Run strategies for selected strategy KPI override
    strategy_results, _ = simulate_all(financial_preds)
    selected_strategy = None
    if strategy_filter:
        for s in strategy_results:
            if s['id'] == strategy_filter:
                selected_strategy = s
                break

    # Flat-stake P/L on value bets
    result_map = {'Home Win': 'H', 'Draw': 'D', 'Away Win': 'A'}
    value_financial_preds = [p for p in financial_preds if p.max_value and p.max_value > 0]
    total_staked = 0
    cumulative_pl_values = []
    cumulative_pl_labels = []
    running_pl = 0.0
    peak_pl = 0.0
    max_drawdown = 0.0
    longest_losing_streak = 0
    current_losing_streak = 0

    # Threshold analysis
    threshold_levels = [0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20]
    threshold_results = {t: {
        'staked': 0, 'profit': 0.0, 'wins': 0,
        'peak': 0.0, 'max_drawdown': 0.0,
        'losing_streak': 0, 'max_losing_streak': 0,
        'winning_streak': 0, 'max_winning_streak': 0,
        'odds_sum': 0.0, 'best_win': 0.0, 'worst_loss': 0.0,
    } for t in threshold_levels}
    threshold_cumulative = {t: {'running': 0.0, 'values': []} for t in threshold_levels}

    bet_odds_list = []
    value_bet_wins = 0
    value_total = len(value_financial_preds)

    for p in value_financial_preds:
        bet_outcome = result_map.get(p.max_value_result)
        if not bet_outcome:
            continue

        odds_map = {'H': p.odds_h, 'D': p.odds_d, 'A': p.odds_a}
        bet_odds = odds_map.get(bet_outcome)
        if not bet_odds or bet_odds <= 1:
            prob_map = {'H': p.h_win, 'D': p.draw, 'A': p.a_win}
            model_prob = prob_map.get(bet_outcome, 0)
            if model_prob <= 0:
                continue
            bet_odds = 1.0 / model_prob

        won = (bet_outcome == p.actual_ftr)
        if won:
            value_bet_wins += 1
        bet_odds_list.append(bet_odds)

        total_staked += 1
        if won:
            running_pl += (bet_odds - 1)
            current_losing_streak = 0
        else:
            running_pl -= 1
            current_losing_streak += 1
            longest_losing_streak = max(longest_losing_streak, current_losing_streak)

        if running_pl > peak_pl:
            peak_pl = running_pl
        drawdown = peak_pl - running_pl
        if drawdown > max_drawdown:
            max_drawdown = drawdown

        cumulative_pl_values.append(round(running_pl, 2))
        cumulative_pl_labels.append(p.date.strftime('%d %b'))

        for t in threshold_levels:
            if p.max_value >= t:
                r = threshold_results[t]
                r['staked'] += 1
                r['odds_sum'] += bet_odds
                if won:
                    payout = bet_odds - 1
                    r['profit'] += payout
                    r['wins'] += 1
                    r['losing_streak'] = 0
                    r['winning_streak'] += 1
                    r['max_winning_streak'] = max(r['max_winning_streak'], r['winning_streak'])
                    if payout > r['best_win']:
                        r['best_win'] = payout
                else:
                    r['profit'] -= 1
                    r['winning_streak'] = 0
                    r['losing_streak'] += 1
                    r['max_losing_streak'] = max(r['max_losing_streak'], r['losing_streak'])
                if r['profit'] > r['peak']:
                    r['peak'] = r['profit']
                dd = r['peak'] - r['profit']
                if dd > r['max_drawdown']:
                    r['max_drawdown'] = dd
                threshold_cumulative[t]['running'] = round(r['profit'], 2)
            threshold_cumulative[t]['values'].append(threshold_cumulative[t]['running'])

    total_profit = round(running_pl, 2)
    roi = round(total_profit / total_staked * 100, 1) if total_staked > 0 else 0

    # Build threshold table
    threshold_table = []
    for t in threshold_levels:
        r = threshold_results[t]
        t_roi = round(r['profit'] / r['staked'] * 100, 1) if r['staked'] > 0 else 0
        t_winrate = round(r['wins'] / r['staked'] * 100, 1) if r['staked'] > 0 else 0
        t_avg_odds = round(r['odds_sum'] / r['staked'], 2) if r['staked'] > 0 else 0
        if t_avg_odds > 1 and r['staked'] > 0:
            b = t_avg_odds - 1
            p_w = r['wins'] / r['staked']
            t_kelly = round((b * p_w - (1 - p_w)) / b * 100, 1) if b > 0 else 0
        else:
            t_kelly = 0
        threshold_table.append({
            'threshold': f"{t*100:.0f}%",
            'bets': r['staked'],
            'wins': r['wins'],
            'losses': r['staked'] - r['wins'],
            'win_rate': t_winrate,
            'profit': round(r['profit'], 2),
            'roi': t_roi,
            'avg_odds': t_avg_odds,
            'max_drawdown': round(r['max_drawdown'], 2),
            'max_losing_streak': r['max_losing_streak'],
            'max_winning_streak': r['max_winning_streak'],
            'best_win': round(r['best_win'], 2),
            'kelly': t_kelly,
        })

    # Kelly criterion
    avg_odds = round(sum(bet_odds_list) / len(bet_odds_list), 2) if bet_odds_list else 0
    if avg_odds > 1 and value_total > 0:
        b = avg_odds - 1
        p_win = value_bet_wins / value_total
        q_lose = 1 - p_win
        kelly_fraction = round((b * p_win - q_lose) / b * 100, 1) if b > 0 else 0
    else:
        kelly_fraction = 0

    # Override KPIs if strategy selected
    if selected_strategy:
        display_staked = selected_strategy['bets']
        display_profit = selected_strategy['profit']
        display_roi = selected_strategy['roi']
        display_drawdown = selected_strategy['max_drawdown']
        display_losing_streak = selected_strategy['max_losing_streak']
    else:
        display_staked = total_staked
        display_profit = total_profit
        display_roi = roi
        display_drawdown = round(max_drawdown, 2)
        display_losing_streak = longest_losing_streak

    return {
        'league_filter': league_filter,
        'league_options': league_options,
        'strategy_filter': strategy_filter,
        'strategy_options': strategy_options,
        'total_staked': display_staked,
        'total_profit': display_profit,
        'roi': display_roi,
        'max_drawdown': display_drawdown,
        'longest_losing_streak': display_losing_streak,
        'threshold_table': threshold_table,
        'kelly_fraction': kelly_fraction,
        'avg_odds': round(avg_odds, 2),
        'strategy_chart_json': json.dumps({
            'labels': cumulative_pl_labels,
            'datasets': [
                {
                    'label': f">{t*100:.0f}%",
                    'data': threshold_cumulative[t]['values'],
                }
                for t in threshold_levels
            ],
        }),
    }


def _get_strategies_context(request):
    """Build context for the betting strategies page."""
    qs, league_filter, league_options = _get_base_performance_qs(request)

    from apps.home.strategies import simulate_all

    financial_qs = qs.order_by('date')
    financial_preds = list(financial_qs.only(
        'date', 'div', 'home_team', 'away_team',
        'h_win', 'draw', 'a_win',
        'odds_h', 'odds_d', 'odds_a',
        'max_value', 'max_value_result', 'actual_ftr',
    ))

    strategy_results, strategy_date_labels = simulate_all(financial_preds)

    # Per-league strategy breakdown
    league_preds = defaultdict(list)
    for p in financial_preds:
        league_preds[p.div].append(p)

    league_strategy_data = []
    for div in sorted(league_preds.keys()):
        preds = league_preds[div]
        lg_results, _ = simulate_all(preds)
        league_strategy_data.append({
            'code': div,
            'name': LEAGUE_NAMES.get(div, div),
            'strategies': lg_results,
        })
    league_strategy_data.sort(key=lambda x: x['name'])

    return {
        'league_filter': league_filter,
        'league_options': league_options,
        'betting_strategies': strategy_results,
        'league_strategy_data': league_strategy_data,
        'betting_strategy_chart_json': json.dumps({
            'labels': strategy_date_labels,
            'datasets': [
                {
                    'label': s['name'],
                    'data': s['pl_values'],
                    'color': s['color'],
                }
                for s in strategy_results
            ],
        }),
    }


@login_required(login_url="/login/")
def methodology(request):
    """Static page explaining the prediction methodology."""
    context = {'segment': 'methodology'}
    return HttpResponse(loader.get_template('home/methodology.html').render(context, request))


@login_required(login_url="/login/")
def performance(request):
    """Redirect old /performance/ URL to the accuracy sub-page."""
    query_string = request.META.get('QUERY_STRING', '')
    url = reverse('performance_accuracy')
    if query_string:
        url += '?' + query_string
    return HttpResponseRedirect(url)


@login_required(login_url="/login/")
def performance_accuracy(request):
    context = {'segment': 'performance_accuracy'}
    context.update(_get_accuracy_context(request))
    html_template = loader.get_template('home/performance_accuracy.html')
    return HttpResponse(html_template.render(context, request))


@login_required(login_url="/login/")
def performance_financial(request):
    context = {'segment': 'performance_financial'}
    context.update(_get_financial_context(request))
    html_template = loader.get_template('home/performance_financial.html')
    return HttpResponse(html_template.render(context, request))


@login_required(login_url="/login/")
def performance_strategies(request):
    context = {'segment': 'performance_strategies'}
    context.update(_get_strategies_context(request))
    html_template = loader.get_template('home/performance_strategies.html')
    return HttpResponse(html_template.render(context, request))


def _get_btts_accuracy_context(request):
    """Build context for the BTTS accuracy page."""
    qs, league_filter, league_options = _get_base_performance_qs(request)

    # Only include predictions that have actual goals (needed to determine BTTS)
    filtered_qs = qs.filter(actual_fthg__isnull=False, actual_ftag__isnull=False)

    team_filter = request.GET.get('team', '').strip()
    result_filter = request.GET.get('result', '')

    if team_filter:
        filtered_qs = filtered_qs.filter(Q(home_team__icontains=team_filter) | Q(away_team__icontains=team_filter))

    total_resolved = filtered_qs.count()

    # For each prediction: model predicted BTTS (btts_yes > 0.5) vs actual BTTS (both scored)
    correct_count = 0
    btts_pred_yes_total = 0
    btts_pred_yes_correct = 0
    calibration_buckets = defaultdict(lambda: {'count': 0, 'correct': 0})
    league_stats_raw = defaultdict(lambda: {'total': 0, 'correct': 0})
    team_stats_raw = defaultdict(lambda: {'matches': 0, 'correct': 0})
    time_stats_raw = defaultdict(lambda: {'total': 0, 'correct': 0})
    results_data = []

    for p in filtered_qs.only(
        'btts_yes', 'actual_fthg', 'actual_ftag', 'div',
        'home_team', 'away_team', 'date', 'pred_fthg', 'pred_ftag',
    ):
        actual_btts = (p.actual_fthg > 0 and p.actual_ftag > 0)
        pred_btts = (p.btts_yes > 0.5) if p.btts_yes else False
        is_correct = (pred_btts == actual_btts)

        if is_correct:
            correct_count += 1
        if pred_btts:
            btts_pred_yes_total += 1
            if actual_btts:
                btts_pred_yes_correct += 1

        # Calibration: bucket by btts_yes probability
        if p.btts_yes is not None:
            bucket = min(int(p.btts_yes * 10), 9)
            calibration_buckets[bucket]['count'] += 1
            if actual_btts:
                calibration_buckets[bucket]['correct'] += 1

        # League stats
        league_stats_raw[p.div]['total'] += 1
        if is_correct:
            league_stats_raw[p.div]['correct'] += 1

        # Team stats
        for team in [p.home_team, p.away_team]:
            team_stats_raw[team]['matches'] += 1
            if is_correct:
                team_stats_raw[team]['correct'] += 1

        # Time stats
        month_key = p.date.strftime('%Y-%m')
        time_stats_raw[month_key]['total'] += 1
        if is_correct:
            time_stats_raw[month_key]['correct'] += 1

    # Apply result filter for the results table
    if result_filter == 'correct':
        table_preds = []
        for p in filtered_qs.order_by('-date', 'div'):
            actual_btts = (p.actual_fthg > 0 and p.actual_ftag > 0)
            pred_btts = (p.btts_yes > 0.5) if p.btts_yes else False
            if pred_btts == actual_btts:
                table_preds.append(p)
    elif result_filter == 'incorrect':
        table_preds = []
        for p in filtered_qs.order_by('-date', 'div'):
            actual_btts = (p.actual_fthg > 0 and p.actual_ftag > 0)
            pred_btts = (p.btts_yes > 0.5) if p.btts_yes else False
            if pred_btts != actual_btts:
                table_preds.append(p)
    else:
        table_preds = list(filtered_qs.order_by('-date', 'div'))

    overall_accuracy = round(correct_count / total_resolved * 100, 1) if total_resolved > 0 else 0
    btts_yes_accuracy = round(btts_pred_yes_correct / btts_pred_yes_total * 100, 1) if btts_pred_yes_total > 0 else 0

    # Format league stats
    league_stats = []
    for div, stats in league_stats_raw.items():
        league_stats.append({
            'name': LEAGUE_NAMES.get(div, div),
            'total': stats['total'],
            'correct': stats['correct'],
            'accuracy': round(stats['correct'] / stats['total'] * 100, 1) if stats['total'] > 0 else 0,
        })
    league_stats.sort(key=lambda x: x['accuracy'], reverse=True)

    # Format team stats
    team_stats_list = []
    for team, stats in sorted(team_stats_raw.items()):
        stats['team'] = team
        stats['accuracy'] = round(stats['correct'] / stats['matches'] * 100, 1) if stats['matches'] > 0 else 0
        team_stats_list.append(stats)
    team_stats_list.sort(key=lambda x: x['accuracy'], reverse=True)

    # Calibration chart data
    cal_labels = []
    cal_predicted = []
    cal_actual = []
    for i in range(10):
        cal_labels.append(f"{i*10}-{(i+1)*10}%")
        cal_predicted.append((i * 10 + (i + 1) * 10) / 2)
        bucket_data = calibration_buckets[i]
        if bucket_data['count'] > 0:
            cal_actual.append(round(bucket_data['correct'] / bucket_data['count'] * 100, 1))
        else:
            cal_actual.append(None)

    # Time chart data
    time_labels = []
    time_accuracy = []
    for month_key in sorted(time_stats_raw.keys()):
        stats = time_stats_raw[month_key]
        from datetime import datetime as dt
        time_labels.append(dt.strptime(month_key, '%Y-%m').strftime('%b %Y'))
        time_accuracy.append(round(stats['correct'] / stats['total'] * 100, 1) if stats['total'] > 0 else 0)

    # Paginate results
    paginator = Paginator(table_preds, 25)
    page_obj = paginator.get_page(request.GET.get('page', 1))

    return {
        'total_resolved': total_resolved,
        'overall_accuracy': overall_accuracy,
        'correct_count': correct_count,
        'btts_pred_yes_total': btts_pred_yes_total,
        'btts_yes_accuracy': btts_yes_accuracy,
        'league_stats': league_stats,
        'team_stats': team_stats_list,
        'cal_labels_json': json.dumps(cal_labels),
        'cal_predicted_json': json.dumps(cal_predicted),
        'cal_actual_json': json.dumps(cal_actual),
        'time_labels_json': json.dumps(time_labels),
        'time_accuracy_json': json.dumps(time_accuracy),
        'results_page': page_obj,
        'league_filter': league_filter,
        'team_filter': team_filter,
        'result_filter': result_filter,
        'league_options': league_options,
    }


def _get_ou_accuracy_context(request):
    """Build context for the Over/Under 2.5 accuracy page."""
    qs, league_filter, league_options = _get_base_performance_qs(request)

    filtered_qs = qs.filter(actual_fthg__isnull=False, actual_ftag__isnull=False)

    team_filter = request.GET.get('team', '').strip()
    result_filter = request.GET.get('result', '')

    if team_filter:
        filtered_qs = filtered_qs.filter(Q(home_team__icontains=team_filter) | Q(away_team__icontains=team_filter))

    total_resolved = filtered_qs.count()

    correct_count = 0
    over_pred_total = 0
    over_pred_correct = 0
    calibration_buckets = defaultdict(lambda: {'count': 0, 'correct': 0})
    league_stats_raw = defaultdict(lambda: {'total': 0, 'correct': 0})
    team_stats_raw = defaultdict(lambda: {'matches': 0, 'correct': 0})
    time_stats_raw = defaultdict(lambda: {'total': 0, 'correct': 0})

    # Value bucket analysis for O/U
    value_buckets_def = [
        ('<0%', -999, 0),
        ('0-5%', 0, 0.05),
        ('5-10%', 0.05, 0.10),
        ('10-15%', 0.10, 0.15),
        ('15%+', 0.15, 999),
    ]
    value_bucket_data = {label: {'count': 0, 'correct': 0} for label, _, _ in value_buckets_def}

    for p in filtered_qs.only(
        'over25', 'under25', 'over25_value', 'under25_value',
        'actual_fthg', 'actual_ftag', 'div',
        'home_team', 'away_team', 'date',
    ):
        actual_over = ((p.actual_fthg + p.actual_ftag) > 2)
        pred_over = (p.over25 > 0.5) if p.over25 else False
        is_correct = (pred_over == actual_over)

        if is_correct:
            correct_count += 1
        if pred_over:
            over_pred_total += 1
            if actual_over:
                over_pred_correct += 1

        # Calibration: bucket by over25 probability
        if p.over25 is not None:
            bucket = min(int(p.over25 * 10), 9)
            calibration_buckets[bucket]['count'] += 1
            if actual_over:
                calibration_buckets[bucket]['correct'] += 1

        # League stats
        league_stats_raw[p.div]['total'] += 1
        if is_correct:
            league_stats_raw[p.div]['correct'] += 1

        # Team stats
        for team in [p.home_team, p.away_team]:
            team_stats_raw[team]['matches'] += 1
            if is_correct:
                team_stats_raw[team]['correct'] += 1

        # Time stats
        month_key = p.date.strftime('%Y-%m')
        time_stats_raw[month_key]['total'] += 1
        if is_correct:
            time_stats_raw[month_key]['correct'] += 1

        # Value bucket analysis (use max of over25_value, under25_value)
        ou_max_val = max(p.over25_value or -999, p.under25_value or -999)
        for label, low, high in value_buckets_def:
            if low <= ou_max_val < high:
                value_bucket_data[label]['count'] += 1
                if is_correct:
                    value_bucket_data[label]['correct'] += 1
                break

    # Apply result filter for the results table
    if result_filter == 'correct':
        table_preds = [
            p for p in filtered_qs.order_by('-date', 'div')
            if ((p.over25 > 0.5) if p.over25 else False) == ((p.actual_fthg + p.actual_ftag) > 2)
        ]
    elif result_filter == 'incorrect':
        table_preds = [
            p for p in filtered_qs.order_by('-date', 'div')
            if ((p.over25 > 0.5) if p.over25 else False) != ((p.actual_fthg + p.actual_ftag) > 2)
        ]
    else:
        table_preds = list(filtered_qs.order_by('-date', 'div'))

    overall_accuracy = round(correct_count / total_resolved * 100, 1) if total_resolved > 0 else 0
    over_pred_accuracy = round(over_pred_correct / over_pred_total * 100, 1) if over_pred_total > 0 else 0

    # Format league stats
    league_stats = []
    for div, stats in league_stats_raw.items():
        league_stats.append({
            'name': LEAGUE_NAMES.get(div, div),
            'total': stats['total'],
            'correct': stats['correct'],
            'accuracy': round(stats['correct'] / stats['total'] * 100, 1) if stats['total'] > 0 else 0,
        })
    league_stats.sort(key=lambda x: x['accuracy'], reverse=True)

    # Format team stats
    team_stats_list = []
    for team, stats in sorted(team_stats_raw.items()):
        stats['team'] = team
        stats['accuracy'] = round(stats['correct'] / stats['matches'] * 100, 1) if stats['matches'] > 0 else 0
        team_stats_list.append(stats)
    team_stats_list.sort(key=lambda x: x['accuracy'], reverse=True)

    # Calibration chart data (for Over 2.5 probability)
    cal_labels = []
    cal_predicted = []
    cal_actual = []
    for i in range(10):
        cal_labels.append(f"{i*10}-{(i+1)*10}%")
        cal_predicted.append((i * 10 + (i + 1) * 10) / 2)
        bucket_data = calibration_buckets[i]
        if bucket_data['count'] > 0:
            cal_actual.append(round(bucket_data['correct'] / bucket_data['count'] * 100, 1))
        else:
            cal_actual.append(None)

    # Value bucket chart data
    vb_labels = []
    vb_accuracy = []
    vb_counts = []
    for label, _, _ in value_buckets_def:
        bd = value_bucket_data[label]
        vb_labels.append(label)
        vb_accuracy.append(round(bd['correct'] / bd['count'] * 100, 1) if bd['count'] > 0 else 0)
        vb_counts.append(bd['count'])

    # Time chart data
    time_labels = []
    time_accuracy = []
    for month_key in sorted(time_stats_raw.keys()):
        stats = time_stats_raw[month_key]
        from datetime import datetime as dt
        time_labels.append(dt.strptime(month_key, '%Y-%m').strftime('%b %Y'))
        time_accuracy.append(round(stats['correct'] / stats['total'] * 100, 1) if stats['total'] > 0 else 0)

    paginator = Paginator(table_preds, 25)
    page_obj = paginator.get_page(request.GET.get('page', 1))

    return {
        'total_resolved': total_resolved,
        'overall_accuracy': overall_accuracy,
        'correct_count': correct_count,
        'over_pred_total': over_pred_total,
        'over_pred_accuracy': over_pred_accuracy,
        'league_stats': league_stats,
        'team_stats': team_stats_list,
        'cal_labels_json': json.dumps(cal_labels),
        'cal_predicted_json': json.dumps(cal_predicted),
        'cal_actual_json': json.dumps(cal_actual),
        'value_bucket_labels_json': json.dumps(vb_labels),
        'value_bucket_accuracy_json': json.dumps(vb_accuracy),
        'value_bucket_counts_json': json.dumps(vb_counts),
        'time_labels_json': json.dumps(time_labels),
        'time_accuracy_json': json.dumps(time_accuracy),
        'results_page': page_obj,
        'league_filter': league_filter,
        'team_filter': team_filter,
        'result_filter': result_filter,
        'league_options': league_options,
    }


def _get_ou_financial_context(request):
    """Build context for the Over/Under 2.5 financial analysis page."""
    qs, league_filter, league_options = _get_base_performance_qs(request)
    strategy_filter = request.GET.get('strategy', '')

    from apps.home.ou_strategies import simulate_ou_all, OU_STRATEGIES
    strategy_options = [{'id': s['id'], 'name': s['name']} for s in OU_STRATEGIES]

    financial_qs = qs.filter(
        actual_fthg__isnull=False, actual_ftag__isnull=False,
    ).order_by('date')

    financial_preds = list(financial_qs.only(
        'date', 'div', 'home_team', 'away_team',
        'over25', 'under25', 'over25_value', 'under25_value',
        'odds_over25', 'odds_under25',
        'actual_fthg', 'actual_ftag',
    ))

    # Run O/U strategies for strategy filter dropdown
    ou_strategy_results, _ = simulate_ou_all(financial_preds)
    selected_strategy = None
    if strategy_filter:
        for s in ou_strategy_results:
            if s['id'] == strategy_filter:
                selected_strategy = s
                break

    # Flat-stake P/L on O/U value bets
    # A "value bet" is when max(over25_value, under25_value) > 0
    total_staked = 0
    running_pl = 0.0
    peak_pl = 0.0
    max_drawdown = 0.0
    longest_losing_streak = 0
    current_losing_streak = 0
    cumulative_pl_values = []
    cumulative_pl_labels = []
    bet_odds_list = []
    value_bet_wins = 0

    # Threshold analysis
    threshold_levels = [0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20]
    threshold_results = {t: {
        'staked': 0, 'profit': 0.0, 'wins': 0,
        'peak': 0.0, 'max_drawdown': 0.0,
        'losing_streak': 0, 'max_losing_streak': 0,
        'winning_streak': 0, 'max_winning_streak': 0,
        'odds_sum': 0.0, 'best_win': 0.0,
    } for t in threshold_levels}
    threshold_cumulative = {t: {'running': 0.0, 'values': []} for t in threshold_levels}

    for p in financial_preds:
        over_val = p.over25_value or -999
        under_val = p.under25_value or -999
        ou_max_val = max(over_val, under_val)
        if ou_max_val <= 0:
            # Not a value bet — still need to update threshold cumulative for chart continuity
            for t in threshold_levels:
                threshold_cumulative[t]['values'].append(threshold_cumulative[t]['running'])
            continue

        # Determine which side to bet
        if over_val >= under_val:
            bet_over = True
            bet_odds = p.odds_over25
        else:
            bet_over = False
            bet_odds = p.odds_under25

        if not bet_odds or bet_odds <= 1:
            for t in threshold_levels:
                threshold_cumulative[t]['values'].append(threshold_cumulative[t]['running'])
            continue

        actual_total = (p.actual_fthg or 0) + (p.actual_ftag or 0)
        actual_over = actual_total > 2
        won = (bet_over and actual_over) or (not bet_over and not actual_over)

        if won:
            value_bet_wins += 1
        bet_odds_list.append(bet_odds)

        total_staked += 1
        if won:
            running_pl += (bet_odds - 1)
            current_losing_streak = 0
        else:
            running_pl -= 1
            current_losing_streak += 1
            longest_losing_streak = max(longest_losing_streak, current_losing_streak)

        if running_pl > peak_pl:
            peak_pl = running_pl
        drawdown = peak_pl - running_pl
        if drawdown > max_drawdown:
            max_drawdown = drawdown

        cumulative_pl_values.append(round(running_pl, 2))
        cumulative_pl_labels.append(p.date.strftime('%d %b'))

        for t in threshold_levels:
            if ou_max_val >= t:
                r = threshold_results[t]
                r['staked'] += 1
                r['odds_sum'] += bet_odds
                if won:
                    payout = bet_odds - 1
                    r['profit'] += payout
                    r['wins'] += 1
                    r['losing_streak'] = 0
                    r['winning_streak'] += 1
                    r['max_winning_streak'] = max(r['max_winning_streak'], r['winning_streak'])
                    if payout > r['best_win']:
                        r['best_win'] = payout
                else:
                    r['profit'] -= 1
                    r['winning_streak'] = 0
                    r['losing_streak'] += 1
                    r['max_losing_streak'] = max(r['max_losing_streak'], r['losing_streak'])
                if r['profit'] > r['peak']:
                    r['peak'] = r['profit']
                dd = r['peak'] - r['profit']
                if dd > r['max_drawdown']:
                    r['max_drawdown'] = dd
                threshold_cumulative[t]['running'] = round(r['profit'], 2)
            threshold_cumulative[t]['values'].append(threshold_cumulative[t]['running'])

    total_profit = round(running_pl, 2)
    roi = round(total_profit / total_staked * 100, 1) if total_staked > 0 else 0

    # Build threshold table
    threshold_table = []
    for t in threshold_levels:
        r = threshold_results[t]
        t_roi = round(r['profit'] / r['staked'] * 100, 1) if r['staked'] > 0 else 0
        t_winrate = round(r['wins'] / r['staked'] * 100, 1) if r['staked'] > 0 else 0
        t_avg_odds = round(r['odds_sum'] / r['staked'], 2) if r['staked'] > 0 else 0
        if t_avg_odds > 1 and r['staked'] > 0:
            b = t_avg_odds - 1
            p_w = r['wins'] / r['staked']
            t_kelly = round((b * p_w - (1 - p_w)) / b * 100, 1) if b > 0 else 0
        else:
            t_kelly = 0
        threshold_table.append({
            'threshold': f"{t*100:.0f}%",
            'bets': r['staked'],
            'wins': r['wins'],
            'losses': r['staked'] - r['wins'],
            'win_rate': t_winrate,
            'profit': round(r['profit'], 2),
            'roi': t_roi,
            'avg_odds': t_avg_odds,
            'max_drawdown': round(r['max_drawdown'], 2),
            'max_losing_streak': r['max_losing_streak'],
            'max_winning_streak': r['max_winning_streak'],
            'best_win': round(r['best_win'], 2),
            'kelly': t_kelly,
        })

    avg_odds = round(sum(bet_odds_list) / len(bet_odds_list), 2) if bet_odds_list else 0
    if avg_odds > 1 and total_staked > 0:
        b = avg_odds - 1
        p_win = value_bet_wins / total_staked
        kelly_fraction = round((b * p_win - (1 - p_win)) / b * 100, 1) if b > 0 else 0
    else:
        kelly_fraction = 0

    # Override KPIs if strategy selected
    if selected_strategy:
        display_staked = selected_strategy['bets']
        display_profit = selected_strategy['profit']
        display_roi = selected_strategy['roi']
        display_drawdown = selected_strategy['max_drawdown']
        display_losing_streak = selected_strategy['max_losing_streak']
    else:
        display_staked = total_staked
        display_profit = total_profit
        display_roi = roi
        display_drawdown = round(max_drawdown, 2)
        display_losing_streak = longest_losing_streak

    return {
        'league_filter': league_filter,
        'league_options': league_options,
        'strategy_filter': strategy_filter,
        'strategy_options': strategy_options,
        'total_staked': display_staked,
        'total_profit': display_profit,
        'roi': display_roi,
        'max_drawdown': display_drawdown,
        'longest_losing_streak': display_losing_streak,
        'threshold_table': threshold_table,
        'kelly_fraction': kelly_fraction,
        'avg_odds': avg_odds,
        'strategy_chart_json': json.dumps({
            'labels': cumulative_pl_labels,
            'datasets': [
                {
                    'label': f">{t*100:.0f}%",
                    'data': threshold_cumulative[t]['values'],
                }
                for t in threshold_levels
            ],
        }),
    }


def _get_ou_strategies_context(request):
    """Build context for the O/U betting strategies page."""
    qs, league_filter, league_options = _get_base_performance_qs(request)

    from apps.home.ou_strategies import simulate_ou_all

    financial_qs = qs.filter(
        actual_fthg__isnull=False, actual_ftag__isnull=False,
    ).order_by('date')
    financial_preds = list(financial_qs.only(
        'date', 'div', 'home_team', 'away_team',
        'over25', 'under25', 'over25_value', 'under25_value',
        'odds_over25', 'odds_under25',
        'actual_fthg', 'actual_ftag',
    ))

    strategy_results, strategy_date_labels = simulate_ou_all(financial_preds)

    # Per-league strategy breakdown
    league_preds = defaultdict(list)
    for p in financial_preds:
        league_preds[p.div].append(p)

    league_strategy_data = []
    for div in sorted(league_preds.keys()):
        preds = league_preds[div]
        lg_results, _ = simulate_ou_all(preds)
        league_strategy_data.append({
            'code': div,
            'name': LEAGUE_NAMES.get(div, div),
            'strategies': lg_results,
        })
    league_strategy_data.sort(key=lambda x: x['name'])

    return {
        'league_filter': league_filter,
        'league_options': league_options,
        'betting_strategies': strategy_results,
        'league_strategy_data': league_strategy_data,
        'betting_strategy_chart_json': json.dumps({
            'labels': strategy_date_labels,
            'datasets': [
                {
                    'label': s['name'],
                    'data': s['pl_values'],
                    'color': s['color'],
                }
                for s in strategy_results
            ],
        }),
    }


@login_required(login_url="/login/")
def performance_ou_strategies(request):
    context = {'segment': 'performance_ou_strategies'}
    context.update(_get_ou_strategies_context(request))
    html_template = loader.get_template('home/performance_ou_strategies.html')
    return HttpResponse(html_template.render(context, request))


@login_required(login_url="/login/")
def performance_btts_accuracy(request):
    context = {'segment': 'performance_btts_accuracy'}
    context.update(_get_btts_accuracy_context(request))
    html_template = loader.get_template('home/performance_btts_accuracy.html')
    return HttpResponse(html_template.render(context, request))


@login_required(login_url="/login/")
def performance_ou_accuracy(request):
    context = {'segment': 'performance_ou_accuracy'}
    context.update(_get_ou_accuracy_context(request))
    html_template = loader.get_template('home/performance_ou_accuracy.html')
    return HttpResponse(html_template.render(context, request))


@login_required(login_url="/login/")
def performance_ou_financial(request):
    context = {'segment': 'performance_ou_financial'}
    context.update(_get_ou_financial_context(request))
    html_template = loader.get_template('home/performance_ou_financial.html')
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
