"""Agent creation and execution using Claude Agent SDK."""

import asyncio
import importlib
import os
import sys

from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, ResultMessage, TextBlock, ToolUseBlock

from agents.config import AgentConfig, get_agent_config
from agents.hooks import create_hooks


def _get_project_root():
    """Get the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _get_system_prompt(config: AgentConfig) -> str:
    """Load the system prompt for an agent from its prompt module."""
    module = importlib.import_module(config.system_prompt_module)
    prompt = module.PROMPT

    # Append working directory context
    prompt += f"\n\n## Working Directory\n{_get_project_root()}\n"
    return prompt


def _build_options(config: AgentConfig, permission_mode: str = 'default') -> ClaudeAgentOptions:
    """Build ClaudeAgentOptions for an agent."""
    options_kwargs = {
        'system_prompt': _get_system_prompt(config),
        'permission_mode': permission_mode,
        'cwd': _get_project_root(),
        'allowed_tools': config.tools,
    }

    # Add custom MCP tools for pipeline and review agents
    if config.agent_type in ('pipeline', 'review'):
        try:
            from agents.tools.mcp_server import create_tools_server
            server = create_tools_server()
            options_kwargs['mcp_servers'] = {'football-value-tools': server}
            # Add MCP tool names to allowed tools
            mcp_tools = [
                'mcp__football-value-tools__get_prediction_stats',
                'mcp__football-value-tools__get_recent_predictions',
                'mcp__football-value-tools__get_team_performance',
                'mcp__football-value-tools__validate_predictions_csv',
                'mcp__football-value-tools__check_data_source_availability',
            ]
            options_kwargs['allowed_tools'] = config.tools + mcp_tools
        except ImportError:
            pass  # SDK not installed, skip MCP tools

    # Add safety hooks
    options_kwargs['hooks'] = create_hooks(config)

    return ClaudeAgentOptions(**options_kwargs)


async def run_agent_async(config: AgentConfig, task: str,
                          permission_mode: str = 'default',
                          verbose: bool = False) -> str:
    """Run an agent asynchronously and return collected output."""
    options = _build_options(config, permission_mode)

    output_parts = []

    async for message in query(prompt=task, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if hasattr(block, 'text'):
                    text = block.text
                    output_parts.append(text)
                    if verbose:
                        print(text)
                elif hasattr(block, 'name'):
                    if verbose:
                        print(f"[Tool] {block.name}")
        elif isinstance(message, ResultMessage):
            if hasattr(message, 'result') and message.result:
                output_parts.append(str(message.result))
                if verbose:
                    print(f"\n{'='*60}")
                    print(message.result)

    return "\n".join(output_parts) if output_parts else "Agent completed without output."


def run_agent(config: AgentConfig, task: str,
              permission_mode: str = 'default',
              verbose: bool = False) -> str:
    """Run an agent synchronously. Entry point for CLI and management commands."""
    return asyncio.run(run_agent_async(config, task, permission_mode, verbose))
