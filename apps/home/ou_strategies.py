"""Over/Under 2.5 betting strategy definitions and financial simulation engine.

Each strategy defines a filter that decides whether to bet on a given prediction,
and which O/U side to bet on. The simulate_ou_all() function runs all strategies
in a single pass over the data for efficiency.
"""

from collections import defaultdict

OU_STRATEGIES = [
    {
        'id': 'all_ou_value',
        'name': 'All O/U Value Bets',
        'description': 'Bet on the best O/U side whenever max(over25_value, under25_value) > 0',
        'color': '#1d8cf8',
    },
    {
        'id': 'high_ou_value',
        'name': 'High O/U Value Filter',
        'description': 'Only bet when O/U model edge exceeds 10%',
        'color': '#00f2c3',
    },
    {
        'id': 'strong_over',
        'name': 'Strong Over',
        'description': 'Value bet where over25 probability > 55% and over is the value pick',
        'color': '#ff8d72',
    },
    {
        'id': 'under_specialist',
        'name': 'Under Specialist',
        'description': 'Only bet Under 2.5 when under25_value > 0',
        'color': '#e14eca',
    },
    {
        'id': 'over_specialist',
        'name': 'Over Specialist',
        'description': 'Only bet Over 2.5 when over25_value > 0',
        'color': '#fd5d93',
    },
    {
        'id': 'ou_confidence',
        'name': 'O/U Confidence Picks',
        'description': 'Value + decisive probability + predictable teams/league + sensible odds + model-market agreement',
        'color': '#ba54f5',
    },
]


def _get_ou_odds(p, bet_over):
    """Get bookmaker odds for Over or Under, with model fallback."""
    if bet_over:
        if p.odds_over25 and p.odds_over25 > 1:
            return p.odds_over25
        if p.over25 and p.over25 > 0:
            return 1.0 / p.over25
    else:
        if p.odds_under25 and p.odds_under25 > 1:
            return p.odds_under25
        if p.under25 and p.under25 > 0:
            return 1.0 / p.under25
    return None


def _ou_max_value(p):
    """Return (max_value, bet_over) for the best O/U side."""
    ov = p.over25_value if p.over25_value else 0
    uv = p.under25_value if p.under25_value else 0
    if ov >= uv:
        return ov, True
    return uv, False


def _evaluate_ou_strategies(p, team_tracker=None):
    """Evaluate all O/U strategies for a single prediction.

    Returns a list of (bet_over: bool, bet_odds: float) or None for each strategy.
    None means the strategy skips this prediction.
    """
    results = []
    ou_max_val, best_is_over = _ou_max_value(p)

    # Strategy 1: All O/U Value Bets
    if ou_max_val > 0:
        odds = _get_ou_odds(p, best_is_over)
        results.append((best_is_over, odds) if odds else None)
    else:
        results.append(None)

    # Strategy 2: High O/U Value Filter - edge > 10%
    if ou_max_val > 0.10:
        odds = _get_ou_odds(p, best_is_over)
        results.append((best_is_over, odds) if odds else None)
    else:
        results.append(None)

    # Strategy 3: Strong Over - over25_value > 0, over25 > 55%, over is value pick
    ov = p.over25_value if p.over25_value else 0
    if ov > 0 and p.over25 and p.over25 > 0.55 and best_is_over:
        odds = _get_ou_odds(p, True)
        results.append((True, odds) if odds else None)
    else:
        results.append(None)

    # Strategy 4: Under Specialist - only under25_value > 0
    uv = p.under25_value if p.under25_value else 0
    if uv > 0:
        odds = _get_ou_odds(p, False)
        results.append((False, odds) if odds else None)
    else:
        results.append(None)

    # Strategy 5: Over Specialist - only over25_value > 0
    ov = p.over25_value if p.over25_value else 0
    if ov > 0:
        odds = _get_ou_odds(p, True)
        results.append((True, odds) if odds else None)
    else:
        results.append(None)

    # Strategy 6: O/U Confidence Picks - multi-signal composite
    if ou_max_val > 0 and team_tracker is not None:
        bet_prob = p.over25 if best_is_over else p.under25
        other_prob = p.under25 if best_is_over else p.over25
        bet_prob = bet_prob or 0
        other_prob = other_prob or 0

        # 1. Decisive probability: bet side > 55% and margin >= 10pp
        margin = bet_prob - other_prob
        decisive = (bet_prob > 0.55 and margin >= 0.10)

        # 2. Team O/U predictability: both teams >= 60% O/U accuracy, min 5 matches
        home_stats = team_tracker.get('__ou__' + p.home_team, {'correct': 0, 'total': 0})
        away_stats = team_tracker.get('__ou__' + p.away_team, {'correct': 0, 'total': 0})
        min_team_matches = 5
        min_team_accuracy = 0.60
        home_ok = (home_stats['total'] >= min_team_matches and
                   home_stats['correct'] / home_stats['total'] >= min_team_accuracy)
        away_ok = (away_stats['total'] >= min_team_matches and
                   away_stats['correct'] / away_stats['total'] >= min_team_accuracy)

        # 3. League O/U predictability: >= 50% accuracy, min 20 matches
        league_stats = team_tracker.get('__ou_league__' + p.div, {'correct': 0, 'total': 0})
        min_league_matches = 20
        min_league_accuracy = 0.50
        league_ok = (league_stats['total'] >= min_league_matches and
                     league_stats['correct'] / league_stats['total'] >= min_league_accuracy)

        # 4. Odds range filter: O/U odds tend to be tighter
        odds = _get_ou_odds(p, best_is_over)
        odds_ok = (odds is not None and 1.4 <= odds <= 3.5)

        # 5. O/U value bet win rate per team (min 3 bets, >= 40%)
        home_vb = team_tracker.get('__ou_vb__' + p.home_team)
        away_vb = team_tracker.get('__ou_vb__' + p.away_team)
        min_vb_matches = 3
        min_vb_winrate = 0.40
        home_vb_total = home_vb['total'] if home_vb else 0
        away_vb_total = away_vb['total'] if away_vb else 0
        home_vb_ok = (home_vb_total < min_vb_matches or
                      home_vb.get('wins', 0) / home_vb_total >= min_vb_winrate)
        away_vb_ok = (away_vb_total < min_vb_matches or
                      away_vb.get('wins', 0) / away_vb_total >= min_vb_winrate)

        if (decisive and home_ok and away_ok and league_ok and
                odds_ok and home_vb_ok and away_vb_ok):
            results.append((best_is_over, odds))
        else:
            results.append(None)
    else:
        results.append(None)

    return results


def simulate_ou_all(predictions):
    """Run all O/U strategies over a list of Prediction objects (ordered by date).

    Returns a list of strategy result dicts and chart date labels.
    """
    n = len(OU_STRATEGIES)

    states = [{
        'staked': 0, 'wins': 0, 'profit': 0.0,
        'peak': 0.0, 'max_drawdown': 0.0,
        'losing_streak': 0, 'max_losing_streak': 0,
        'winning_streak': 0, 'max_winning_streak': 0,
        'odds_sum': 0.0, 'best_win': 0.0,
        'pl_values': [],
        'running_pl': 0.0,
    } for _ in range(n)]

    team_tracker = defaultdict(lambda: {'correct': 0, 'total': 0})
    date_labels = []

    for p in predictions:
        # Skip if we don't have actual goals
        if p.actual_fthg is None or p.actual_ftag is None:
            continue

        actual_total = p.actual_fthg + p.actual_ftag
        actual_over = actual_total > 2

        evals = _evaluate_ou_strategies(p, team_tracker=team_tracker)
        any_bet = False

        for i, ev in enumerate(evals):
            if ev is None:
                states[i]['pl_values'].append(round(states[i]['running_pl'], 2))
                continue

            any_bet = True
            bet_over, bet_odds = ev
            won = (bet_over == actual_over)
            s = states[i]

            s['staked'] += 1
            s['odds_sum'] += bet_odds

            if won:
                payout = bet_odds - 1
                s['profit'] += payout
                s['wins'] += 1
                s['losing_streak'] = 0
                s['winning_streak'] += 1
                s['max_winning_streak'] = max(s['max_winning_streak'], s['winning_streak'])
                if payout > s['best_win']:
                    s['best_win'] = payout
            else:
                s['profit'] -= 1
                s['winning_streak'] = 0
                s['losing_streak'] += 1
                s['max_losing_streak'] = max(s['max_losing_streak'], s['losing_streak'])

            if s['profit'] > s['peak']:
                s['peak'] = s['profit']
            dd = s['peak'] - s['profit']
            if dd > s['max_drawdown']:
                s['max_drawdown'] = dd

            s['running_pl'] = s['profit']
            s['pl_values'].append(round(s['running_pl'], 2))

        # Update rolling trackers AFTER evaluation
        # O/U prediction accuracy per team (did over25>0.5 match actual?)
        pred_over = (p.over25 or 0) > 0.5
        ou_correct = (pred_over == actual_over)

        for team in [p.home_team, p.away_team]:
            ou_key = '__ou__' + team
            team_tracker[ou_key]['total'] += 1
            if ou_correct:
                team_tracker[ou_key]['correct'] += 1

        # League O/U accuracy
        league_key = '__ou_league__' + p.div
        team_tracker[league_key]['total'] += 1
        if ou_correct:
            team_tracker[league_key]['correct'] += 1

        # Per-team O/U value bet win rate
        ou_max_val, best_is_over = _ou_max_value(p)
        if ou_max_val > 0:
            vb_won = (best_is_over == actual_over)
            for team in [p.home_team, p.away_team]:
                vb_key = '__ou_vb__' + team
                team_tracker[vb_key]['total'] += 1
                if vb_won:
                    team_tracker[vb_key]['wins'] = team_tracker[vb_key].get('wins', 0) + 1

        if any_bet:
            date_labels.append(p.date.strftime('%d %b'))

    # Build result dicts
    strategy_results = []
    for i, meta in enumerate(OU_STRATEGIES):
        s = states[i]
        win_rate = round(s['wins'] / s['staked'] * 100, 1) if s['staked'] > 0 else 0
        roi = round(s['profit'] / s['staked'] * 100, 1) if s['staked'] > 0 else 0
        avg_odds = round(s['odds_sum'] / s['staked'], 2) if s['staked'] > 0 else 0

        kelly = 0
        if avg_odds > 1 and s['staked'] > 0:
            b = avg_odds - 1
            p_w = s['wins'] / s['staked']
            kelly = round((b * p_w - (1 - p_w)) / b * 100, 1) if b > 0 else 0

        strategy_results.append({
            'id': meta['id'],
            'name': meta['name'],
            'description': meta['description'],
            'color': meta['color'],
            'bets': s['staked'],
            'wins': s['wins'],
            'losses': s['staked'] - s['wins'],
            'win_rate': win_rate,
            'profit': round(s['profit'], 2),
            'roi': roi,
            'max_drawdown': round(s['max_drawdown'], 2),
            'max_losing_streak': s['max_losing_streak'],
            'max_winning_streak': s['max_winning_streak'],
            'best_win': round(s['best_win'], 2),
            'avg_odds': avg_odds,
            'kelly': kelly,
            'pl_values': s['pl_values'],
        })

    return strategy_results, date_labels
