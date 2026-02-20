import pandas as pd
from scipy.stats import poisson
import numpy as np


def analysis(Train, Test):
    """Run Poisson distribution analysis on training data to predict Test fixtures."""
    teams = pd.DataFrame(sorted(Train.HomeTeam.unique()))
    home_form = []
    away_form = []

    for team in teams[0]:
        HF = []
        AF = []

        HF.append(len(Train[Train['HomeTeam'] == team]))
        HF.append(sum(Train['FTHG'][Train['HomeTeam'] == team]))
        HF.append(Train['FTHG'][Train['HomeTeam'] == team].mean())
        HF.append(sum(Train['FTAG'][Train['HomeTeam'] == team]))
        HF.append(Train['FTAG'][Train['HomeTeam'] == team].mean())

        AF.append(len(Train[Train['AwayTeam'] == team]))
        AF.append(sum(Train['FTAG'][Train['AwayTeam'] == team]))
        AF.append(Train['FTAG'][Train['AwayTeam'] == team].mean())
        AF.append(sum(Train['FTHG'][Train['AwayTeam'] == team]))
        AF.append(Train['FTHG'][Train['AwayTeam'] == team].mean())

        home_form.append(HF)
        away_form.append(AF)

    home_form = pd.DataFrame(home_form, columns=[
        'GamesplayedH', 'GoalsforH', 'AvggoalsforH', 'GoalsagainstH', 'AvggoalsagainstH',
    ])
    away_form = pd.DataFrame(away_form, columns=[
        'GamesplayedA', 'GoalsforA', 'AvggoalsforA', 'GoalsagainstA', 'AvggoalsagainstA',
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
    phg = []
    pag = []

    for index, row in Test.iterrows():
        scores = np.zeros((10, 10))
        for score1 in range(0, 10):
            for score2 in range(0, 10):
                scores[score1, score2] = (
                    poisson.pmf(score1, row['HTGE']) * poisson.pmf(score2, row['ATGE'])
                )

        draw.append(sum(np.diag(scores)))
        HWin.append(sum(sum(np.tril(scores))) - sum(np.diag(scores)))
        AWin.append(sum(sum(np.triu(scores))) - sum(np.diag(scores)))

        homeg, awayg = np.where(scores == np.amax(scores))
        phg.append(homeg[0])
        pag.append(awayg[0])

    Test['HWin'] = HWin
    Test['Draw'] = draw
    Test['AWin'] = AWin
    Test['Pred_FTHG'] = phg
    Test['Pred_FTAG'] = pag
    Test['Pred_FTR'] = np.nan
    Test['Pred_FTR'] = (
        Test['Pred_FTR']
        .mask(Test.Pred_FTHG == Test.Pred_FTAG, "D")
        .mask(Test.Pred_FTHG > Test.Pred_FTAG, "H")
        .mask(Test.Pred_FTHG < Test.Pred_FTAG, "A")
    )

    return Test
