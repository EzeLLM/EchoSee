from langchain_core.tools import tool
import os
import dotenv
dotenv.load_dotenv()
from logger import logger
logger = logger.Logger('agent')


from tavily import TavilyClient
@tool
def search(query:str, time_range:str,return_raw_results) -> str:
    """
    Call to search the web for the query, returns the response.
    Should be used to learn recent news, new things, etc.
    return_raw_results: bool, if True, returns the raw results from the web, else returns the answer.
    query: string, query to ask the web. Query must be google search style query, not a chat style question.
    time_range: string, time range to search in. can be 'day', 'week', 'month', 'year' or 'none'.
    returns: string, response from the web.
    """
    
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
    



if __name__ == '__main__':
    # Test the search function
    query = "Who killed JFK?"
    time_range = "month"

    try:
        # Use invoke instead of direct call
        result = search.invoke({"query": query, "time_range": time_range, "return_raw_results": True})
        print(f"Search Results for '{query}' in the past '{time_range}':\n{result}")
    except Exception as e:
        print(f"An error occurred: {e}")
