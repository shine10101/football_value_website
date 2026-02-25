import pandas as pd
import numpy as np


def value(predictions):
    """Calculate value by comparing model probabilities against average market odds."""
    # Use average market odds (AvgH/D/A) with B365 as fallback
    for avg_col, b365_col, target in [('AvgH', 'B365H', 'Odds_H'), ('AvgD', 'B365D', 'Odds_D'), ('AvgA', 'B365A', 'Odds_A')]:
        if avg_col in predictions.columns:
            predictions[target] = pd.to_numeric(predictions[avg_col], errors='coerce')
        elif b365_col in predictions.columns:
            predictions[target] = pd.to_numeric(predictions[b365_col], errors='coerce')
        else:
            predictions[target] = np.nan

    odds = predictions[['Odds_H', 'Odds_D', 'Odds_A']]
    implied_odds = 1 / odds
    probability = predictions[['HWin', 'Draw', 'AWin']]
    val = pd.DataFrame(
        probability.values - implied_odds.values,
        columns=['Home Win', 'Draw', 'Away Win'],
    )
    predictions['Max_Value'] = val.max(axis=1).values
    predictions['Max_Value_Result'] = val.idxmax(axis=1).values

    # Store the market odds used for the best-value result
    result_odds_map = {'Home Win': 'Odds_H', 'Draw': 'Odds_D', 'Away Win': 'Odds_A'}
    predictions['Best_Odds'] = predictions.apply(
        lambda r: r[result_odds_map.get(r['Max_Value_Result'], 'Odds_H')], axis=1
    )

    # Over/Under 2.5 value using average market odds
    if 'Avg>2.5' in predictions.columns and 'Avg<2.5' in predictions.columns:
        over_odds = pd.to_numeric(predictions['Avg>2.5'], errors='coerce')
        under_odds = pd.to_numeric(predictions['Avg<2.5'], errors='coerce')
        predictions['Over25_Value'] = predictions['Over25'] - (1 / over_odds)
        predictions['Under25_Value'] = predictions['Under25'] - (1 / under_odds)
        predictions['Odds_Over25'] = over_odds
        predictions['Odds_Under25'] = under_odds
    else:
        predictions['Over25_Value'] = np.nan
        predictions['Under25_Value'] = np.nan
        predictions['Odds_Over25'] = np.nan
        predictions['Odds_Under25'] = np.nan

    # Best O/U value: pick whichever of Over/Under has higher value
    ou_vals = predictions[['Over25_Value', 'Under25_Value']]
    predictions['OU_Max_Value'] = ou_vals.max(axis=1)
    predictions['OU_Best_Bet'] = ou_vals.idxmax(axis=1).map({
        'Over25_Value': 'Over 2.5',
        'Under25_Value': 'Under 2.5',
    })
    predictions['OU_Best_Odds'] = predictions.apply(
        lambda r: r['Odds_Over25'] if r['OU_Best_Bet'] == 'Over 2.5' else r['Odds_Under25'], axis=1
    )

    predictions = predictions.sort_values('Max_Value', ascending=False)
    return predictions
