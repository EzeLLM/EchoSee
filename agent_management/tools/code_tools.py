"""Code-related tools for the agent (LeetCode, etc.)."""

from langchain_core.tools import tool
from agent_management.helpers.leetcode.agent import client as leetcode_client


@tool
def leetcode_agent(task: str) -> str:
    """LeetCode agent to solve problems or provide hints about them.

    Parameters:
        task: The task to solve (e.g., "solve problem 141" or "give me hints for problem 75")

    Returns:
        The solution to the task with the problem context
    """
    return leetcode_client.run(task)
