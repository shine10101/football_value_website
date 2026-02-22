"""Betting strategy definitions and financial simulation engine.

Each strategy defines a filter that decides whether to bet on a given prediction,
and which outcome to bet on. The simulate_all() function runs all strategies
in a single pass over the data for efficiency.
"""

from collections import defaultdict

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
    {
        'id': 'confidence_picks',
        'name': 'Confidence Picks',
        'description': 'Value + decisive probability + predictable teams/league + sensible odds + model-market agreement',
        'color': '#ba54f5',
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


def _evaluate_strategies(p, team_tracker=None):
    """Evaluate all strategies for a single prediction.

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

    # Strategy 6: Confidence Picks - multi-signal composite filter
    # Requires: value, decisive probability, model-market agreement,
    #           predictable teams & league, sensible odds range
    if p.max_value > 0 and bet_outcome and team_tracker is not None:
        probs = {'H': p.h_win or 0, 'D': p.draw or 0, 'A': p.a_win or 0}
        bet_prob = probs.get(bet_outcome, 0)

        # 1. Model-market agreement: value pick matches most-likely outcome
        max_prob_outcome = max(probs, key=probs.get)
        agreement = (max_prob_outcome == bet_outcome)

        # 2. Probability margin: bet outcome must lead by >= 10pp over next best
        sorted_probs = sorted(probs.values(), reverse=True)
        margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) >= 2 else 0
        decisive = (bet_prob > 0.50 and margin >= 0.10)

        # 3. Team predictability: both teams >= 60% accuracy, min 5 matches
        home_stats = team_tracker.get(p.home_team, {'correct': 0, 'total': 0})
        away_stats = team_tracker.get(p.away_team, {'correct': 0, 'total': 0})
        min_team_matches = 5
        min_team_accuracy = 0.50
        home_ok = (home_stats['total'] >= min_team_matches and
                   home_stats['correct'] / home_stats['total'] >= min_team_accuracy)
        away_ok = (away_stats['total'] >= min_team_matches and
                   away_stats['correct'] / away_stats['total'] >= min_team_accuracy)

        # 4. League predictability: league >= 50% accuracy, min 20 matches
        league_stats = team_tracker.get('__league__' + p.div, {'correct': 0, 'total': 0})
        min_league_matches = 20
        min_league_accuracy = 0.40
        league_ok = (league_stats['total'] >= min_league_matches and
                     league_stats['correct'] / league_stats['total'] >= min_league_accuracy)

        # 5. Odds range filter: avoid extreme longshots and tiny payouts
        odds = _get_odds(p, bet_outcome)
        odds_ok = (odds is not None and 1.4 <= odds <= 4.0)

        # 6. Value bet win rate per team: track if the value outcome
        #    actually happened historically for these teams (min 3 bets)
        home_vb = team_tracker.get('__vb__' + p.home_team)
        away_vb = team_tracker.get('__vb__' + p.away_team)
        min_vb_matches = 3
        min_vb_winrate = 0.40
        home_vb_total = home_vb['total'] if home_vb else 0
        away_vb_total = away_vb['total'] if away_vb else 0
        home_vb_ok = (home_vb_total < min_vb_matches or
                      home_vb.get('wins', 0) / home_vb_total >= min_vb_winrate)
        away_vb_ok = (away_vb_total < min_vb_matches or
                      away_vb.get('wins', 0) / away_vb_total >= min_vb_winrate)

        if (agreement and decisive and home_ok and away_ok and
                league_ok and odds_ok and home_vb_ok and away_vb_ok):
            results.append((bet_outcome, odds))
        else:
            results.append(None)
    else:
        results.append(None)

    return results


def simulate_all(predictions):
    """Run all strategies over a list of Prediction objects (ordered by date).

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

    # Rolling team accuracy tracker for confidence picks strategy
    team_tracker = defaultdict(lambda: {'correct': 0, 'total': 0})

    # Shared x-axis labels (from baseline strategy - most bets)
    date_labels = []

    for p in predictions:
        evals = _evaluate_strategies(p, team_tracker=team_tracker)
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

        # Update rolling trackers AFTER evaluating (so current match
        # doesn't influence its own decision)
        is_correct = (p.pred_ftr == p.actual_ftr)

        # Team prediction accuracy
        team_tracker[p.home_team]['total'] += 1
        team_tracker[p.away_team]['total'] += 1
        if is_correct:
            team_tracker[p.home_team]['correct'] += 1
            team_tracker[p.away_team]['correct'] += 1

        # League prediction accuracy
        league_key = '__league__' + p.div
        team_tracker[league_key]['total'] += 1
        if is_correct:
            team_tracker[league_key]['correct'] += 1

        # Per-team value bet win rate (did the value outcome actually happen?)
        if p.max_value and p.max_value > 0:
            vb_outcome = RESULT_MAP.get(p.max_value_result)
            vb_won = (vb_outcome == p.actual_ftr)
            for team in [p.home_team, p.away_team]:
                vb_key = '__vb__' + team
                team_tracker[vb_key]['total'] += 1
                if vb_won:
                    team_tracker[vb_key]['wins'] = team_tracker[vb_key].get('wins', 0) + 1

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
