from django.core.management.base import BaseCommand

from apps.home.predictions.pipeline import run_predictions


class Command(BaseCommand):
    help = 'Run the full prediction pipeline: fetch data, resolve past results, generate new predictions'

    def handle(self, *args, **options):
        self.stdout.write('Running full prediction pipeline...')
        pred = run_predictions()

        if pred.empty:
            self.stdout.write(self.style.WARNING('No predictions generated'))
        else:
            self.stdout.write(self.style.SUCCESS(f'Generated {len(pred)} predictions'))
