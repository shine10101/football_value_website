import pandas as pd


def value(predictions):
    """Calculate value by comparing model probabilities against bookmaker odds."""
    odds = predictions[['B365H', 'B365D', 'B365A']]
    implied_odds = 1 / odds
    probability = predictions[['HWin', 'Draw', 'AWin']]
    val = pd.DataFrame(
        probability.values - implied_odds.values,
        columns=['Home Win', 'Draw', 'Away Win'],
    )
    predictions['Max_Value'] = val.max(axis=1).values
    predictions['Max_Value_Result'] = val.idxmax(axis=1).values
    predictions = predictions.sort_values('Max_Value', ascending=False)
    return predictions
