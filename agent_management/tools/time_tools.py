"""Time-related tools for the agent."""

from langchain_core.tools import tool
from datetime import datetime


@tool
def get_current_time() -> str:
    """Gets the current time in text format.

    Returns:
        Current time as string in format "YYYY-MM-DD HH:MM:SS"
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def get_current_date() -> str:
    """Gets the current date in text format.

    Returns:
        Current date as string in format "YYYY-MM-DD"
    """
    return datetime.now().strftime("%Y-%m-%d")
