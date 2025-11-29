from langchain_core.tools import tool
import os
import dotenv
dotenv.load_dotenv()
from logger import logger
logger = logger.Logger('agent')
from utils.utils import litellm_llm, llm
from core.clients import clients
from datetime import datetime
from agent_management.helpers.leetcode.agent import client as leetcode_client
from agent_management.helpers.search.SearchClient import search_agent as _robust_search_agent




@tool
# get the current time in text format
def get_current_time() -> str:
    """
    Gets the current time in text format.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
# get the current date in text format
def get_current_date() -> str:
    """
    Gets the current date in text format.
    """
    return datetime.now().strftime("%Y-%m-%d")




# NOTE : should i replace the tavily search with duckduckgo search?
@tool
def search(query:str, time_range:str,return_raw_results) -> str:
    """
 Searches the web for the specified query using the Tavily API and returns the response.
    
    This function connects to the Tavily search service to retrieve up-to-date information
    about recent events, news, or other current topics.
    Should be used for simple tasks like 'what is the weather in tokyo' or 'what is the capital of france' etc. but not for complex tasks like 'what is the current leader of the christian world' or 'what is the latest news on the war in ukraine' etc.
    
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
        If return_raw_results is False, returns an answer, which may not be so accurate."""
    
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
    print(f"Searching for '{query}' (time range: '{time_range}'), return raw: {return_raw_results}")

    response = tavily_client.search(**search_params)

    # Process results
    if return_raw_results:
        result = []
        for item in response.get('results', []):
            result.append(f"Title: {item['title']}\nURL: {item['url']}\nContent: {item['content']}\n\n")
        return ''.join(result).strip()
    
    return response.get('answer', 'No answer available')
    
# @tool
# def compute(code:str) -> int:
#     """
#     Gets python code and executes it.
#     If you want it to return a value, you must explicitly return the value. also you can import python internal libraries.
#     Parameters:
#         code: The python code to execute.
#     Returns:
#         The result of the code.
#     """
#     return exec(code)


# @tool
# def code_agent(task:str) -> str:
#     """
#     Uses a code agent to solve the task. Should be used for complex tasks that require multiple steps or code execution. Do not use this tool for simple tasks.
#     Parameters:
#         task: The task to solve.
#     Returns:
#         The solution to the task.
#     """
#     agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=litellm_llm)
#     return agent.run(task)

@tool
def leetcode_agent(task:str) -> str:
    """
    LeetCode agent to solve problems or provide hints about them. 
    Parameters:
        task: The task to solve.
    Returns:
        The solution to the task with the problem context.
    """
    return leetcode_client.run(task)

# ---------------------------------------------------------------------------
# Diversified fact-checking search agent (see helpers/search/SearchClient.py)
# ---------------------------------------------------------------------------


@tool
def robust_search(query: str, max_queries: int = 3) -> str:
    """
    Perform a diversified web search to gather evidence from multiple sources
    and produce a concise, fact-checked answer.

    Should be used if the user asks for 'detailed search' or tells you to 'gather everything you can find' etc.
    The agent should not be used for simple tasks like 'what is the weather in tokyo' or 'what is the capital of france' etc. but for querries like 'what is the current leader of the christian world' or 'what is the latest news on the war in ukraine' etc.

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

if __name__ == '__main__':
    # Test the search function
    query = "JFK documents"
    time_range = "none"

    try:
        # Use invoke instead of direct call
        result = search.invoke({"query": query, "time_range": time_range, "return_raw_results": False})
        print(f"Search Results for '{query}' in the past '{time_range}':\n{result}")
    except Exception as e:
        print(f"An error occurred: {e}")


    # result = code_agent.invoke({"task": "compute the 3rd factorial of e^12"})
    # print(result)
    
