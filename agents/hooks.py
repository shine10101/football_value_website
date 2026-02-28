"""Safety hooks for agent operations.

Prevents agents from:
- Reading or writing .env, .env.production, or other secret files
- Running destructive shell commands
- Writing to files outside their allowed scope

Uses the Claude Agent SDK hook API:
- Hook callbacks receive (input_data, tool_use_id, context)
- Return a SyncHookJSONOutput dict to block or modify tool calls
"""

import fnmatch

from claude_agent_sdk import HookMatcher


# Files that no agent should access
SECRET_PATTERNS = ['.env', '.env.production', 'credentials', 'secrets']

# Shell commands that no agent should run
DESTRUCTIVE_COMMANDS = [
    'rm -rf',
    'rm db.sqlite3',
    'rm predictions.csv',
    'git push --force',
    'git reset --hard',
    'DROP TABLE',
    'DELETE FROM',
    'format ',
    'del /f',
    'git clean -f',
]


def _deny(reason):
    """Return a hook output that denies the tool call."""
    return {
        'hookSpecificOutput': {
            'hookEventName': 'PreToolUse',
            'permissionDecision': 'deny',
            'permissionDecisionReason': reason,
        }
    }


def _make_secret_file_hook():
    """Create a hook that blocks access to secret files."""
    async def hook(input_data, tool_use_id, context):
        tool_input = input_data.get('tool_input', {})
        path = tool_input.get('file_path', '') or ''
        if not path:
            return {}
        filename = path.replace('\\', '/').split('/')[-1].lower()
        for pattern in SECRET_PATTERNS:
            if pattern in filename:
                return _deny(f"Access to '{filename}' blocked (matches secret pattern '{pattern}')")
        return {}
    return hook


def _make_destructive_command_hook():
    """Create a hook that blocks destructive shell commands."""
    async def hook(input_data, tool_use_id, context):
        tool_input = input_data.get('tool_input', {})
        cmd = tool_input.get('command', '')
        if not cmd:
            return {}
        cmd_lower = cmd.lower()
        for pattern in DESTRUCTIVE_COMMANDS:
            if pattern.lower() in cmd_lower:
                return _deny(f"Destructive command blocked: '{pattern}'")
        return {}
    return hook


def _make_write_restriction_hook(protected_paths):
    """Create a hook that blocks writes to protected file patterns."""
    async def hook(input_data, tool_use_id, context):
        if not protected_paths:
            return {}
        tool_input = input_data.get('tool_input', {})
        path = tool_input.get('file_path', '')
        if not path:
            return {}
        filename = path.replace('\\', '/')
        for pattern in protected_paths:
            if pattern == '*':
                return _deny("This agent has read-only file access")
            if fnmatch.fnmatch(filename, pattern) or fnmatch.fnmatch(filename.split('/')[-1], pattern):
                return _deny(f"Writing to files matching '{pattern}' is not allowed for this agent")
        return {}
    return hook


def _make_bash_allowlist_hook(allowed_commands):
    """Create a hook that only allows bash commands matching the allowlist."""
    async def hook(input_data, tool_use_id, context):
        if not allowed_commands:
            return {}  # Empty allowlist means full access
        tool_input = input_data.get('tool_input', {})
        cmd = tool_input.get('command', '').strip()
        for allowed in allowed_commands:
            if cmd.startswith(allowed):
                return {}  # Allowed
        return _deny(
            f"Command not in allowlist. Allowed prefixes: {allowed_commands}"
        )
    return hook


def create_hooks(agent_config):
    """Create HookMatcher list for PreToolUse based on agent configuration.

    Returns a dict suitable for ClaudeAgentOptions hooks parameter.
    """
    matchers = []

    # Block secret file access for Read/Write/Edit
    matchers.append(HookMatcher(
        matcher='Read|Write|Edit',
        hooks=[_make_secret_file_hook()],
    ))

    # Block destructive bash commands
    matchers.append(HookMatcher(
        matcher='Bash',
        hooks=[_make_destructive_command_hook()],
    ))

    # Per-agent write restrictions
    if agent_config.protected_paths:
        matchers.append(HookMatcher(
            matcher='Write|Edit',
            hooks=[_make_write_restriction_hook(agent_config.protected_paths)],
        ))

    # Per-agent bash allowlist
    if agent_config.bash_allowed_commands:
        matchers.append(HookMatcher(
            matcher='Bash',
            hooks=[_make_bash_allowlist_hook(agent_config.bash_allowed_commands)],
        ))

    return {'PreToolUse': matchers}
