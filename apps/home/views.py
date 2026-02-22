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
