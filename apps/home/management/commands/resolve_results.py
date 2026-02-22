from django.core.management.base import BaseCommand

from apps.home.predictions import data_ingestion
from apps.home.predictions.pipeline import resolve_results


class Command(BaseCommand):
    help = 'Resolve unresolved predictions with actual match results from football-data.co.uk'

    def handle(self, *args, **options):
        self.stdout.write('Fetching historical data...')
        _data, data_dct = data_ingestion.get_links()

        self.stdout.write('Resolving predictions...')
        count = resolve_results(data_dct)

        self.stdout.write(self.style.SUCCESS(f'Resolved {count} predictions'))
