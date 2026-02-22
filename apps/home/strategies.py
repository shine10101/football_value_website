"""Betting strategy definitions and financial simulation engine.

Each strategy defines a filter that decides whether to bet on a given prediction,
and which outcome to bet on. The simulate_all() function runs all strategies
in a single pass over the data for efficiency.
"""

RESULT_MAP = {'Home Win': 'H', 'Draw': 'D', 'Away Win': 'A'}
OUTCOME_TO_RESULT = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}

STRATEGIES = [
    {
        'id': 'all_value',
        'name': 'All Value Bets',
        'description': 'Bet on the max value outcome whenever value > 0%',
        'color': '#1d8cf8',
    },
    {
        'id': 'high_value',
        'name': 'High Value Filter',
        'description': 'Only bet when model edge exceeds 10%',
        'color': '#00f2c3',
    },
    {
        'id': 'strong_favourites',
        'name': 'Strong Favourites',
        'description': 'Value bet where model probability > 55% and aligns with value pick',
        'color': '#ff8d72',
    },
    {
        'id': 'home_advantage',
        'name': 'Home Advantage',
        'description': 'Only bet home wins with positive value',
        'color': '#e14eca',
    },
    {
        'id': 'draw_specialist',
        'name': 'Draw Specialist',
        'description': 'Bet draws when draw probability exceeds implied odds',
        'color': '#fd5d93',
    },
]


def _get_odds(p, outcome_code):
    """Get bookmaker odds for a given outcome code (H/D/A), with model fallback."""
    odds_map = {'H': p.odds_h, 'D': p.odds_d, 'A': p.odds_a}
    bet_odds = odds_map.get(outcome_code)
    if bet_odds and bet_odds > 1:
        return bet_odds
    # Fallback to model-implied odds
    prob_map = {'H': p.h_win, 'D': p.draw, 'A': p.a_win}
    model_prob = prob_map.get(outcome_code, 0)
    if model_prob > 0:
        return 1.0 / model_prob
    return None


def _evaluate_strategies(p):
    """Evaluate all 5 strategies for a single prediction.

    Returns a list of (outcome_code, bet_odds) or None for each strategy.
    None means the strategy skips this prediction.
    """
    results = []
    bet_outcome = RESULT_MAP.get(p.max_value_result)

    # Strategy 1: All Value Bets - max_value > 0
    if p.max_value > 0 and bet_outcome:
        odds = _get_odds(p, bet_outcome)
        results.append((bet_outcome, odds) if odds else None)
    else:
        results.append(None)

    # Strategy 2: High Value Filter - max_value > 0.10
    if p.max_value > 0.10 and bet_outcome:
        odds = _get_odds(p, bet_outcome)
        results.append((bet_outcome, odds) if odds else None)
    else:
        results.append(None)

    # Strategy 3: Strong Favourites - max_value > 0, max prob > 55%, aligns with value
    if p.max_value > 0 and bet_outcome:
        probs = {'H': p.h_win or 0, 'D': p.draw or 0, 'A': p.a_win or 0}
        max_prob_outcome = max(probs, key=probs.get)
        max_prob = probs[max_prob_outcome]
        if max_prob > 0.55 and max_prob_outcome == bet_outcome:
            odds = _get_odds(p, bet_outcome)
            results.append((bet_outcome, odds) if odds else None)
        else:
            results.append(None)
    else:
        results.append(None)

    # Strategy 4: Home Advantage - max_value > 0, only home wins
    if p.max_value > 0 and p.max_value_result == 'Home Win':
        odds = _get_odds(p, 'H')
        results.append(('H', odds) if odds else None)
    else:
        results.append(None)

    # Strategy 5: Draw Specialist - draw value > 0
    draw_odds = _get_odds(p, 'D')
    if draw_odds and p.draw and draw_odds > 1:
        draw_value = p.draw - (1.0 / draw_odds)
        if draw_value > 0:
            results.append(('D', draw_odds))
        else:
            results.append(None)
    else:
        results.append(None)

    return results


def simulate_all(predictions):
    """Run all 5 strategies over a list of Prediction objects (ordered by date).

    Returns a list of strategy result dicts and chart data for overlay P/L.
    """
    n = len(STRATEGIES)

    # Per-strategy tracking
    states = [{
        'staked': 0, 'wins': 0, 'profit': 0.0,
        'peak': 0.0, 'max_drawdown': 0.0,
        'losing_streak': 0, 'max_losing_streak': 0,
        'winning_streak': 0, 'max_winning_streak': 0,
        'odds_sum': 0.0, 'best_win': 0.0,
        'pl_values': [],
        'running_pl': 0.0,
    } for _ in range(n)]

    # Shared x-axis labels (from baseline strategy - most bets)
    date_labels = []

    for p in predictions:
        evals = _evaluate_strategies(p)
        any_bet = False

        for i, ev in enumerate(evals):
            if ev is None:
                # Carry forward the current P/L
                states[i]['pl_values'].append(round(states[i]['running_pl'], 2))
                continue

            any_bet = True
            outcome_code, bet_odds = ev
            won = (outcome_code == p.actual_ftr)
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

            # Track drawdown
            if s['profit'] > s['peak']:
                s['peak'] = s['profit']
            dd = s['peak'] - s['profit']
            if dd > s['max_drawdown']:
                s['max_drawdown'] = dd

            s['running_pl'] = s['profit']
            s['pl_values'].append(round(s['running_pl'], 2))

        if any_bet:
            date_labels.append(p.date.strftime('%d %b'))

    # Build result dicts
    strategy_results = []
    for i, meta in enumerate(STRATEGIES):
        s = states[i]
        win_rate = round(s['wins'] / s['staked'] * 100, 1) if s['staked'] > 0 else 0
        roi = round(s['profit'] / s['staked'] * 100, 1) if s['staked'] > 0 else 0
        avg_odds = round(s['odds_sum'] / s['staked'], 2) if s['staked'] > 0 else 0

        # Kelly criterion
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
