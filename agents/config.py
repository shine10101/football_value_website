"""Agent configuration management.

Reads ANTHROPIC_API_KEY from environment (.env or shell).
Provides per-agent model and tool configurations.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentConfig:
    name: str
    agent_type: str  # test, review, pipeline, feature
    model: str
    tools: list
    system_prompt_module: str  # e.g. 'agents.prompts.testing'
    bash_allowed_commands: list = field(default_factory=list)
    protected_paths: list = field(default_factory=list)
    max_turns: int = 50


AGENT_CONFIGS = {
    'test': AgentConfig(
        name='Testing Agent',
        agent_type='test',
        model='claude-sonnet-4-20250514',
        tools=['Read', 'Write', 'Edit', 'Glob', 'Grep', 'Bash', 'Task'],
        system_prompt_module='agents.prompts.testing',
        bash_allowed_commands=[
            'python manage.py test',
            'python -m pytest',
            'pip list',
            'git status',
            'git diff',
        ],
        max_turns=80,
    ),
    'review': AgentConfig(
        name='Code Quality Agent',
        agent_type='review',
        model='claude-sonnet-4-20250514',
        tools=['Read', 'Glob', 'Grep', 'Bash', 'Task'],
        system_prompt_module='agents.prompts.code_quality',
        bash_allowed_commands=[
            'python manage.py check',
            'git diff',
            'git log',
            'git status',
            'python -c',
        ],
        protected_paths=['*'],  # No file writes at all
        max_turns=30,
    ),
    'pipeline': AgentConfig(
        name='Pipeline & Data Agent',
        agent_type='pipeline',
        model='claude-sonnet-4-20250514',
        tools=['Read', 'Glob', 'Grep', 'Bash', 'Task'],
        system_prompt_module='agents.prompts.pipeline_data',
        bash_allowed_commands=[
            'python manage.py shell',
            'python manage.py resolve_results',
            'python manage.py backcast',
            'python manage.py refresh_predictions',
            'python manage.py check',
            'git status',
            'git log',
        ],
        protected_paths=['*.py', '*.html', '*.js', '*.css'],
        max_turns=40,
    ),
    'feature': AgentConfig(
        name='Feature Development Agent',
        agent_type='feature',
        model='claude-sonnet-4-20250514',
        tools=['Read', 'Write', 'Edit', 'Glob', 'Grep', 'Bash', 'Task', 'WebSearch'],
        system_prompt_module='agents.prompts.feature_dev',
        bash_allowed_commands=[],  # Full access
        protected_paths=['.env', '.env.production', 'db.sqlite3'],
        max_turns=100,
    ),
}


def get_agent_config(agent_type: str, model_override: Optional[str] = None) -> AgentConfig:
    """Get configuration for an agent type, with optional model override."""
    if agent_type not in AGENT_CONFIGS:
        raise ValueError(f"Unknown agent type: {agent_type}. Choose from: {list(AGENT_CONFIGS.keys())}")
    config = AGENT_CONFIGS[agent_type]
    if model_override:
        config = AgentConfig(
            name=config.name,
            agent_type=config.agent_type,
            model=model_override,
            tools=config.tools,
            system_prompt_module=config.system_prompt_module,
            bash_allowed_commands=config.bash_allowed_commands,
            protected_paths=config.protected_paths,
            max_turns=config.max_turns,
        )
    return config
