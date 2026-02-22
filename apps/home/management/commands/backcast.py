"""Backcast predictions against historical results.

Downloads historical match data, walks through matches chronologically,
and generates predictions using only data available at the time of each match.
Results are saved as resolved Prediction objects for the performance page.

Usage:
    python manage.py backcast
    python manage.py backcast --leagues E0 E1 --season 2526 --min-games 10
"""

import pandas as pd
from collections import defaultdict
from django.core.management.base import BaseCommand

from apps.home.predictions import poisson_analysis, calculate_value
from apps.home.predictions.data_ingestion import _current_season
from apps.home.predictions.pipeline import _safe_float


class Command(BaseCommand):
    help = 'Backcast predictions against historical results to populate performance data'

    def add_arguments(self, parser):
        parser.add_argument(
            '--leagues', nargs='+', default=['E0', 'E1'],
            help='League codes to backcast (default: E0 E1)',
        )
        parser.add_argument(
            '--all', action='store_true', default=False,
            help='Backcast all supported leagues (overrides --leagues)',
        )
        parser.add_argument(
            '--season', type=str, default=None,
            help='Season code e.g. 2526 (default: auto-detect current season)',
        )
        parser.add_argument(
            '--min-games', type=int, default=10,
            help='Minimum games per team before backcasting begins (default: 10)',
        )
        parser.add_argument(
            '--clear', action='store_true', default=False,
            help='Delete existing resolved predictions before backcasting',
        )

    def handle(self, *args, **options):
        from apps.home.models import Prediction

        if options['all']:
            from apps.home.predictions.data_ingestion import LEAGUE_CODES
            leagues = LEAGUE_CODES
        else:
            leagues = options['leagues']

        season = options['season'] or _current_season()
        min_games = options['min_games']

        if options['clear']:
            deleted, _ = Prediction.objects.filter(resolved=True).delete()
            self.stdout.write(self.style.WARNING(f'Cleared {deleted} existing resolved predictions.'))

        if options['all']:
            self.stdout.write(self.style.WARNING(
                f'Backcasting all {len(leagues)} leagues. This may take 15-30 minutes.'
            ))

        total_predicted = 0
        total_correct = 0

        for i, league in enumerate(leagues, 1):
            url = f'https://football-data.co.uk/mmz4281/{season}/{league}.csv'
            self.stdout.write(f'[{i}/{len(leagues)}] Fetching {league} data from {url}...')

            try:
                df = pd.read_csv(url, parse_dates=['Date'], date_format='%d/%m/%Y')
            except Exception as e:
                self.stderr.write(self.style.ERROR(f'Failed to fetch {league}: {e}'))
                continue

            # Ensure required columns exist
            required = {'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR'}
            if not required.issubset(df.columns):
                self.stderr.write(self.style.ERROR(
                    f'{league}: missing required columns. Have: {list(df.columns)}'
                ))
                continue

            has_odds = {'B365H', 'B365D', 'B365A'}.issubset(df.columns)

            # Drop rows without results (unplayed matches at end of CSV)
            df = df.dropna(subset=['FTR', 'FTHG', 'FTAG'])
            df = df.sort_values('Date').reset_index(drop=True)
            df['Div'] = league

            # Track games played per team
            games_played = defaultdict(int)

            # Group by date to process matchdays together
            match_dates = df['Date'].unique()
            league_predicted = 0
            league_correct = 0

            for match_date in match_dates:
                matchday = df[df['Date'] == match_date]
                prior = df[df['Date'] < match_date]

                # Skip if not enough training data
                if len(prior) < 20:
                    # Update game counts for this matchday
                    for _, row in matchday.iterrows():
                        games_played[row['HomeTeam']] += 1
                        games_played[row['AwayTeam']] += 1
                    continue

                # Filter matches where both teams have enough games
                eligible = matchday[matchday.apply(
                    lambda r: (games_played[r['HomeTeam']] >= min_games
                               and games_played[r['AwayTeam']] >= min_games),
                    axis=1,
                )]

                if not eligible.empty:
                    try:
                        # Run Poisson model: train on prior matches, predict eligible
                        # Reset index so calculate_value index alignment works correctly
                        predictions = poisson_analysis.analysis(prior, eligible.reset_index(drop=True))

                        # Calculate value if odds are available
                        if has_odds:
                            predictions = calculate_value.value(predictions)
                        else:
                            predictions['Max_Value'] = 0.0
                            predictions['Max_Value_Result'] = ''

                        # Save each prediction as a resolved Prediction
                        for _, row in predictions.iterrows():
                            try:
                                date_val = pd.to_datetime(row['Date']).date()
                            except Exception:
                                continue

                            pred_ftr = str(row.get('Pred_FTR', ''))
                            if pd.isna(row.get('Pred_FTR')):
                                pred_ftr = ''
                            max_value_result = str(row.get('Max_Value_Result', ''))
                            if pd.isna(row.get('Max_Value_Result')):
                                max_value_result = ''

                            actual_ftr = str(row['FTR'])
                            is_correct = pred_ftr == actual_ftr

                            Prediction.objects.update_or_create(
                                div=league,
                                date=date_val,
                                home_team=str(row['HomeTeam']),
                                away_team=str(row['AwayTeam']),
                                defaults={
                                    'h_win': _safe_float(row.get('HWin')),
                                    'draw': _safe_float(row.get('Draw')),
                                    'a_win': _safe_float(row.get('AWin')),
                                    'pred_ftr': pred_ftr,
                                    'pred_fthg': _safe_float(row.get('Pred_FTHG')),
                                    'pred_ftag': _safe_float(row.get('Pred_FTAG')),
                                    'max_value': _safe_float(row.get('Max_Value')),
                                    'max_value_result': max_value_result,
                                    'odds_h': _safe_float(row.get('B365H')) or None,
                                    'odds_d': _safe_float(row.get('B365D')) or None,
                                    'odds_a': _safe_float(row.get('B365A')) or None,
                                    'actual_ftr': actual_ftr,
                                    'actual_fthg': int(row['FTHG']),
                                    'actual_ftag': int(row['FTAG']),
                                    'resolved': True,
                                },
                            )

                            league_predicted += 1
                            if is_correct:
                                league_correct += 1

                    except Exception as e:
                        self.stderr.write(self.style.WARNING(
                            f'{league} {match_date}: prediction failed - {e}'
                        ))

                # Update game counts after processing this matchday
                for _, row in matchday.iterrows():
                    games_played[row['HomeTeam']] += 1
                    games_played[row['AwayTeam']] += 1

            accuracy = round(league_correct / league_predicted * 100, 1) if league_predicted > 0 else 0
            self.stdout.write(self.style.SUCCESS(
                f'{league}: {league_predicted} predictions, '
                f'{league_correct} correct ({accuracy}%)'
            ))
            total_predicted += league_predicted
            total_correct += league_correct

        overall_accuracy = round(total_correct / total_predicted * 100, 1) if total_predicted > 0 else 0
        self.stdout.write(self.style.SUCCESS(
            f'\nTotal: {total_predicted} predictions, '
            f'{total_correct} correct ({overall_accuracy}%)'
        ))
