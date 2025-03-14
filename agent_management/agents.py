from langchain_core.tools import tool
import os
import dotenv
dotenv.load_dotenv()
from logger import logger
logger = logger.Logger('agent')
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel
from utils.utils import litellm_llm, llm
from tavily import TavilyClient

# NOTE : should i replace the tavily search with duckduckgo search?
@tool
def search(query:str, time_range:str,return_raw_results) -> str:
    """
 Searches the web for the specified query using the Tavily API and returns the response.
    
    This function connects to the Tavily search service to retrieve up-to-date information
    about recent events, news, or other current topics.
    
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
    
    time_range = time_range.lower()
    if time_range not in ['day', 'week', 'month', 'year', 'none']:
        logger.error(f"Invalid time range: {time_range}, setting to 'none'.")
        time_range = 'none'
    client = TavilyClient(os.environ['TAVILY'])
    if return_raw_results:
        response = client.search(
            query=query,
            time_range=time_range
        )
        return response
    response = client.search(
        query=query,
        time_range=time_range,
        include_answer='basic'
    )
    return response['answer'] if 'answer' in response else response['results']
    
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


@tool
def code_agent(task:str) -> str:
    """
    Uses a code agent to solve the task. Should be used for complex tasks that require multiple steps or code execution. Do not use this tool for simple tasks.
    Parameters:
        task: The task to solve.
    Returns:
        The solution to the task.
    """
    agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=litellm_llm)
    return agent.run(task)


if __name__ == '__main__':
    # Test the search function
    # query = "Who killed JFK?"
    # time_range = "month"

    # try:
    #     # Use invoke instead of direct call
    #     result = search.invoke({"query": query, "time_range": time_range, "return_raw_results": True})
    #     print(f"Search Results for '{query}' in the past '{time_range}':\n{result}")
    # except Exception as e:
    #     print(f"An error occurred: {e}")


    result = code_agent.invoke({"task": "compute the 3rd factorial of e^12"})
    print(result)
    
