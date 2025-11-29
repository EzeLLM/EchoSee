"""Web search tools for the agent."""

from langchain_core.tools import tool
from core.clients import clients
from agent_management.helpers.search.SearchClient import search_agent as _robust_search_agent
from logger import logger

logger = logger.Logger('search_tools')


@tool
def search(query: str, time_range: str, return_raw_results: bool) -> str:
    """Searches the web for the specified query using the Tavily API and returns the response.

    This function connects to the Tavily search service to retrieve up-to-date information
    about recent events, news, or other current topics.
    Should be used for simple tasks like 'what is the weather in tokyo' or 'what is the capital of france' etc.
    but not for complex tasks like 'what is the current leader of the christian world' or 'what is the latest news on the war in ukraine' etc.

    Parameters:
        query: The search query to submit to the web search engine.

        time_range: Restricts search results to a specific time period. Valid options are:
                   'day' ,'week' ,'month' ,'year' ,'none'
                   Invalid values will default to 'none' with a warning.

        return_raw_results: Controls the format of the returned data:
                          If True: Returns the search results.
                          If False: Returns only the answer.

    Returns:
        If return_raw_results is True, returns search results.
        If return_raw_results is False, returns an answer, which may not be so accurate.
    """

    # Validate and normalize time_range
    valid_ranges = {'day', 'week', 'month', 'year', 'none'}
    time_range = time_range.lower() if time_range.lower() in valid_ranges else 'none'
    if time_range not in valid_ranges:
        logger.error(f"Invalid time range: {time_range}, defaulting to 'none'")
        time_range = 'none'

    # Configure API parameters
    search_params = {'query': query}
    if time_range != 'none':
        search_params['time_range'] = time_range
    if not return_raw_results:
        search_params['include_answer'] = 'basic'

    # Execute search
    tavily_client = clients.tavily
    logger.info(f"Searching for '{query}' (time range: '{time_range}'), return raw: {return_raw_results}")

    response = tavily_client.search(**search_params)

    # Process results
    if return_raw_results:
        result = []
        for item in response.get('results', []):
            result.append(f"Title: {item['title']}\nURL: {item['url']}\nContent: {item['content']}\n\n")
        return ''.join(result).strip()

    return response.get('answer', 'No answer available')


@tool
def robust_search(query: str, max_queries: int = 3) -> str:
    """Perform a diversified web search to gather evidence from multiple sources
    and produce a concise, fact-checked answer.

    Should be used if the user asks for 'detailed search' or tells you to 'gather everything you can find' etc.
    The agent should not be used for simple tasks like 'what is the weather in tokyo' or 'what is the capital of france' etc.
    but for queries like 'what is the current leader of the christian world' or 'what is the latest news on the war in ukraine' etc.

    This agent:
        1. Generates several alternative search queries using the standard LLM.
        2. Executes those queries via Tavily to obtain raw snippets.
        3. Synthesises the snippets with a high-performance LLM.

    Parameters:
        query:        The user question or topic to investigate.
        max_queries:  Maximum number of diversified search queries to run.

    Returns:
        A short answer that cites sources inline where possible.
    """
    return _robust_search_agent(query, max_queries)
