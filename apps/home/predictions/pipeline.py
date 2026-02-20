import logging

import pandas as pd
from django.conf import settings

from . import data_ingestion, poisson_analysis, calculate_value

logger = logging.getLogger(__name__)


def _fixture_predictions(fixtures, data_dct):
    """Generate predictions for upcoming fixtures using historical league data."""
    leagues = fixtures['Div'].unique()
    predictions = []
    for league in leagues:
        try:
            league_fixtures = fixtures[fixtures['Div'] == league]
            league_data = data_dct[league][0]
            predictions.append(poisson_analysis.analysis(league_data, league_fixtures))
        except Exception:
            logger.warning("Skipping league %s (no historical data)", league)
            continue

    if not predictions:
        return pd.DataFrame()

    return pd.concat(predictions)


def run_predictions():
    """Full pipeline: fetch data, generate predictions, save to CSV.
    Returns the predictions DataFrame."""
    logger.info("Starting prediction pipeline...")

    logger.info("Fetching historical data...")
    _data, data_dct = data_ingestion.get_links()

    logger.info("Fetching upcoming fixtures...")
    fixtures = data_ingestion.get_fixtures()

    logger.info("Generating predictions...")
    pred = _fixture_predictions(fixtures, data_dct)

    if pred.empty:
        logger.warning("No predictions generated")
        return pred

    logger.info("Calculating value...")
    pred = calculate_value.value(pred)

    output_path = settings.PREDICTIONS_CSV
    pred.to_csv(output_path, index=False)
    logger.info("Saved %d predictions to %s", len(pred), output_path)

    return pred
