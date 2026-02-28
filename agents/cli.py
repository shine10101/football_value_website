#!/usr/bin/env python
"""CLI entry point for AI development agents.

Usage:
    python agents/cli.py test "write unit tests for the Poisson analysis module"
    python agents/cli.py review "review views.py for security issues"
    python agents/cli.py pipeline "validate predictions.csv data quality"
    python agents/cli.py feature "split views.py into separate view modules"
    python agents/cli.py test --model claude-opus-4-20250514 "comprehensive tests"

Agent types:
    test     - Testing Agent: writes and runs Django tests
    review   - Code Quality Agent: reviews code for bugs, security, performance
    pipeline - Pipeline & Data Agent: validates data, monitors pipeline health
    feature  - Feature Development Agent: implements features, refactors code
"""

import argparse
import os
import sys

# Add project root to path so Django and agent imports work
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def setup_django():
    """Bootstrap Django settings for agents that need DB access."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
    import django
    django.setup()


def main():
    parser = argparse.ArgumentParser(
        description='Run AI development agents for the football value website',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        'agent',
        choices=['test', 'review', 'pipeline', 'feature'],
        help='Agent type to run',
    )
    parser.add_argument(
        'task',
        help='Task description for the agent (free-form text)',
    )
    parser.add_argument(
        '--model',
        default=None,
        help='Override Claude model (default: claude-sonnet-4-20250514)',
    )
    parser.add_argument(
        '--permission-mode',
        default='default',
        choices=['default', 'acceptEdits', 'bypassPermissions', 'plan'],
        help='Permission mode for the agent (default: default)',
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print agent output in real-time',
    )

    args = parser.parse_args()

    # Check for API key
    if not os.environ.get('ANTHROPIC_API_KEY'):
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        print("Set it in your shell or add it to your .env file.")
        sys.exit(1)

    # Bootstrap Django for agents that need it
    if args.agent in ('pipeline', 'test', 'feature'):
        setup_django()

    from agents.config import get_agent_config
    from agents.definitions import run_agent

    config = get_agent_config(args.agent, model_override=args.model)

    print(f"Starting {config.name}...")
    print(f"  Model: {config.model}")
    print(f"  Task: {args.task}")
    print(f"  Permission mode: {args.permission_mode}")
    print("=" * 60)

    result = run_agent(
        config,
        args.task,
        permission_mode=args.permission_mode,
        verbose=args.verbose,
    )

    if not args.verbose:
        print(result)


if __name__ == '__main__':
    main()
