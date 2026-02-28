"""MCP server exposing custom tools to agents."""

from claude_agent_sdk import create_sdk_mcp_server

from .db_tools import get_prediction_stats, get_recent_predictions, get_team_performance
from .pipeline_tools import validate_predictions_csv, check_data_source_availability


def create_tools_server():
    """Create an MCP server with all custom database and pipeline tools."""
    return create_sdk_mcp_server(
        "football-value-tools",
        tools=[
            get_prediction_stats,
            get_recent_predictions,
            get_team_performance,
            validate_predictions_csv,
            check_data_source_availability,
        ],
    )
