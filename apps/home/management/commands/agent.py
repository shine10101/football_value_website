"""Run AI development agents within the Django context.

Usage:
    python manage.py agent test "write tests for strategies.py"
    python manage.py agent review "check views.py for N+1 queries"
    python manage.py agent pipeline "validate current predictions"
    python manage.py agent feature "add CSV export to performance page"

Shortcuts:
    python manage.py agent pipeline --validate-csv
    python manage.py agent pipeline --check-accuracy --league E0
    python manage.py agent test --coverage
"""

import os
import sys

from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = 'Run AI development agents for testing, review, pipeline monitoring, or feature development'

    def add_arguments(self, parser):
        parser.add_argument(
            'agent_type',
            choices=['test', 'review', 'pipeline', 'feature'],
            help='Type of agent to run',
        )
        parser.add_argument(
            'task',
            nargs='?',
            default='',
            help='Task description (free-form text)',
        )
        parser.add_argument(
            '--model',
            default=None,
            help='Override the Claude model',
        )
        parser.add_argument(
            '--permission-mode',
            default='default',
            choices=['default', 'acceptEdits', 'bypassPermissions', 'plan'],
            help='Permission mode for the agent',
        )
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Print agent output in real-time',
        )
        # Pipeline-specific shortcuts
        parser.add_argument(
            '--validate-csv',
            action='store_true',
            help='Pipeline agent: validate predictions.csv data quality',
        )
        parser.add_argument(
            '--check-accuracy',
            action='store_true',
            help='Pipeline agent: analyze prediction accuracy',
        )
        parser.add_argument(
            '--league',
            type=str,
            default=None,
            help='Filter to specific league code (e.g., E0)',
        )
        # Test-specific shortcuts
        parser.add_argument(
            '--coverage',
            action='store_true',
            help='Test agent: generate tests targeting uncovered code',
        )

    def handle(self, *args, **options):
        # Check for API key
        if not os.environ.get('ANTHROPIC_API_KEY'):
            self.stderr.write(self.style.ERROR(
                'ANTHROPIC_API_KEY environment variable not set. '
                'Set it in your shell or add it to your .env file.'
            ))
            return

        from agents.config import get_agent_config
        from agents.definitions import run_agent

        agent_type = options['agent_type']
        task = options['task']

        # Build task from shortcut flags if no free-form task provided
        if not task:
            if options.get('validate_csv'):
                task = 'Validate predictions.csv for data quality issues. Check probability sums, odds validity, duplicates, and date correctness.'
            elif options.get('check_accuracy'):
                league = options.get('league')
                if league:
                    task = f'Analyze prediction accuracy for league {league}. Show overall accuracy, value bet accuracy, and recent trends.'
                else:
                    task = 'Analyze overall prediction accuracy across all leagues. Show accuracy breakdowns, calibration, and trends.'
            elif options.get('coverage'):
                task = 'Identify untested code in apps/home/ and generate comprehensive tests for maximum coverage. Focus on critical business logic.'
            else:
                self.stderr.write(self.style.ERROR(
                    'Provide a task description or use a shortcut flag '
                    '(--validate-csv, --check-accuracy, --coverage)'
                ))
                return

        config = get_agent_config(agent_type, model_override=options.get('model'))

        self.stdout.write(f"Starting {config.name}...")
        self.stdout.write(f"  Model: {config.model}")
        self.stdout.write(f"  Task: {task}")
        self.stdout.write("=" * 60)

        result = run_agent(
            config,
            task,
            permission_mode=options['permission_mode'],
            verbose=options.get('verbose', False),
        )

        if not options.get('verbose'):
            self.stdout.write(result)
