import pandas as pd
from scipy.stats import poisson
import numpy as np


# Dixon-Coles correlation parameter for low-scoring outcomes.
# Set to 0.0 (disabled) — with single-season training data (~20 games/team),
# the correction adds noise. Re-enable with multi-season data.
DC_RHO = 0.0

# Exponential decay rate per match for recent-form weighting.
# 0.995 gives mild recency bias: a match 100 games ago has weight 0.61.
DECAY_RATE = 0.995


def _dixon_coles_adj(score1, score2, htge, atge, rho):
    """Apply Dixon-Coles correction factor for low-scoring outcomes (0 or 1 goals).

    Returns a multiplier to adjust the independent Poisson probability for
    scorelines where both teams score 0 or 1.
    """
    if score1 == 0 and score2 == 0:
        return 1 - htge * atge * rho
    elif score1 == 0 and score2 == 1:
        return 1 + htge * rho
    elif score1 == 1 and score2 == 0:
        return 1 + atge * rho
    elif score1 == 1 and score2 == 1:
        return 1 - rho
    return 1.0


def _weighted_mean(values, weights):
    """Compute weighted mean, falling back to simple mean if weights sum to 0."""
    w_sum = weights.sum()
    if w_sum == 0:
        return values.mean()
    return (values * weights).sum() / w_sum


def analysis(Train, Test):
    """Run Poisson distribution analysis on training data to predict Test fixtures.

    Improvements over basic Poisson:
    1. Exponential decay weighting — recent matches influence team strength more.
    2. Dixon-Coles adjustment — corrects correlated low-scoring probabilities.
    """
    teams = pd.DataFrame(sorted(Train.HomeTeam.unique()))

    # Assign exponential decay weights: most recent match gets weight 1.0,
    # each older match is multiplied by DECAY_RATE.
    train_sorted = Train.sort_values('Date').reset_index(drop=True)
    n = len(train_sorted)
    weights = pd.Series([DECAY_RATE ** (n - 1 - i) for i in range(n)])
    train_sorted = train_sorted.copy()
    train_sorted['_weight'] = weights.values

    home_form = []
    away_form = []

    for team in teams[0]:
        # Home matches
        mask_h = train_sorted['HomeTeam'] == team
        h_matches = train_sorted[mask_h]
        h_count = len(h_matches)
        if h_count > 0:
            h_goals_for = _weighted_mean(h_matches['FTHG'], h_matches['_weight'])
            h_goals_against = _weighted_mean(h_matches['FTAG'], h_matches['_weight'])
        else:
            h_goals_for = 0.0
            h_goals_against = 0.0

        home_form.append([h_count, h_goals_for, h_goals_against])

        # Away matches
        mask_a = train_sorted['AwayTeam'] == team
        a_matches = train_sorted[mask_a]
        a_count = len(a_matches)
        if a_count > 0:
            a_goals_for = _weighted_mean(a_matches['FTAG'], a_matches['_weight'])
            a_goals_against = _weighted_mean(a_matches['FTHG'], a_matches['_weight'])
        else:
            a_goals_for = 0.0
            a_goals_against = 0.0

        away_form.append([a_count, a_goals_for, a_goals_against])

    home_form = pd.DataFrame(home_form, columns=[
        'GamesplayedH', 'AvggoalsforH', 'AvggoalsagainstH',
    ])
    away_form = pd.DataFrame(away_form, columns=[
        'GamesplayedA', 'AvggoalsforA', 'AvggoalsagainstA',
    ])

    median_avg_H = home_form.median()
    median_avg_A = away_form.median()

    HomeAtkStr = home_form['AvggoalsforH'] / median_avg_H['AvggoalsforH']
    HomeDefStr = home_form['AvggoalsagainstH'] / median_avg_H['AvggoalsagainstH']
    AwayAtkStr = away_form['AvggoalsforA'] / median_avg_A['AvggoalsforA']
    AwayDefStr = away_form['AvggoalsagainstA'] / median_avg_A['AvggoalsagainstA']

    team_strength = pd.concat([HomeAtkStr, HomeDefStr, AwayAtkStr, AwayDefStr], axis=1)
    team_strength.columns = ['HomeAtkStr', 'HomeDefStr', 'AwayAtkStr', 'AwayDefStr']
    team_strength.index = teams[0]

    HTGE = []
    ATGE = []
    for index, row in Test.iterrows():
        HTGE.append(
            team_strength.loc[row['HomeTeam']]['HomeAtkStr']
            * team_strength.loc[row['AwayTeam']]['AwayDefStr']
            * median_avg_H['AvggoalsforH']
        )
        ATGE.append(
            team_strength.loc[row['AwayTeam']]['AwayAtkStr']
            * team_strength.loc[row['HomeTeam']]['HomeDefStr']
            * median_avg_A['AvggoalsforA']
        )

    Test = Test.copy()
    Test['HTGE'] = HTGE
    Test['ATGE'] = ATGE

    draw = []
    HWin = []
    AWin = []
    btts_yes = []
    over25_list = []
    phg = []
    pag = []

    for index, row in Test.iterrows():
        scores = np.zeros((10, 10))
        htge = row['HTGE']
        atge = row['ATGE']

        for score1 in range(10):
            for score2 in range(10):
                p = poisson.pmf(score1, htge) * poisson.pmf(score2, atge)
                # Dixon-Coles adjustment for low-scoring outcomes
                p *= _dixon_coles_adj(score1, score2, htge, atge, DC_RHO)
                scores[score1, score2] = max(p, 0)  # guard against negative

        # Renormalise so the matrix sums to 1.0
        total = scores.sum()
        if total > 0:
            scores /= total

        draw.append(sum(np.diag(scores)))
        HWin.append(sum(sum(np.tril(scores))) - sum(np.diag(scores)))
        AWin.append(sum(sum(np.triu(scores))) - sum(np.diag(scores)))

        # BTTS: probability both teams score at least 1 goal
        btts_yes.append(scores[1:, 1:].sum())

        # Over 2.5 goals: sum of cells where home + away > 2
        over25 = sum(
            scores[i, j]
            for i in range(10) for j in range(10)
            if i + j > 2
        )
        over25_list.append(over25)

        homeg, awayg = np.where(scores == np.amax(scores))
        phg.append(homeg[0])
        pag.append(awayg[0])

    Test['HWin'] = HWin
    Test['Draw'] = draw
    Test['AWin'] = AWin
    Test['BTTS_Yes'] = btts_yes
    Test['BTTS_No'] = [1 - b for b in btts_yes]
    Test['Over25'] = over25_list
    Test['Under25'] = [1 - o for o in over25_list]
    Test['Pred_FTHG'] = phg
    Test['Pred_FTAG'] = pag
    # Determine predicted result from outcome probabilities (not mode scoreline).
    # Using HWin/Draw/AWin is more accurate than comparing Pred_FTHG vs Pred_FTAG
    # because the most-likely scoreline can disagree with the most-likely outcome.
    outcome_probs = pd.DataFrame({'H': HWin, 'D': draw, 'A': AWin})
    Test['Pred_FTR'] = outcome_probs.idxmax(axis=1).values

    return Test
