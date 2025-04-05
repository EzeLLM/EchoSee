from langchain.tools import tool
from langchain.agents import AgentType, initialize_agent
from langchain_openai import ChatOpenAI
import requests
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import os
import json
from bs4 import BeautifulSoup

load_dotenv()

class LeetCodeAPI:
    def __init__(self):
        self.graphql_url = "https://leetcode.com/graphql"
        self.session = requests.Session()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
            "Content-Type": "application/json",
            "Referer": "https://leetcode.com",
            "Origin": "https://leetcode.com"
        }

    def _graphql_query(self, query: str, variables: Dict[str, Any]) -> Optional[Dict]:
        """Execute GraphQL query with error handling"""
        try:
            # Print request details for debugging
            print(f"Sending request to {self.graphql_url}")
            print(f"Headers: {self.headers}")
            print(f"Variables: {variables}")
            
            response = self.session.post(
                self.graphql_url,
                json={"query": query, "variables": variables},
                headers=self.headers,
                timeout=10
            )
            
            # Print response status and headers for debugging
            print(f"Response status: {response.status_code}")
            print(f"Response headers: {response.headers}")
            
            if response.status_code != 200:
                print(f"Error response: {response.text}")
                
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API Request failed: {str(e)}")
            return None
        except json.JSONDecodeError:
            print("Invalid JSON response")
            return None

    def get_problem_data(self, problem_code: str) -> Optional[Dict]:
        """Get problem data using official LeetCode API"""
        # Modified query for public access - simpler approach
        content_query = """
        query questionData($titleSlug: String!) {
            question(titleSlug: $titleSlug) {
                questionId
                title
                difficulty
                content
                topicTags { name }
                codeSnippets { langSlug code }
                exampleTestcases
                metaData
            }
        }
        """

        try:
            # First try to get the problem by ID
            response = requests.get(f"https://leetcode.com/api/problems/all/")
            if response.status_code == 200:
                problems_data = response.json()
                for problem in problems_data.get("stat_status_pairs", []):
                    if str(problem.get("stat", {}).get("question_id")) == problem_code:
                        title_slug = problem.get("stat", {}).get("question__title_slug")
                        break
                else:
                    # If not found by ID, try using the problem code as a slug directly
                    title_slug = problem_code
            else:
                # Fallback: use the problem code as slug
                title_slug = problem_code
        except Exception as e:
            print(f"Error getting title slug: {e}")
            # As a fallback, try to use a mapping of common problems
            common_problems = {
                "1": "two-sum",
                "2": "add-two-numbers",
                # Add more mappings as needed
                "140": "word-break-ii"
            }
            title_slug = common_problems.get(problem_code, problem_code)
        
        print(f"Using title slug: {title_slug}")
        content_vars = {"titleSlug": title_slug}
        
        content_data = self._graphql_query(content_query, content_vars)
        return content_data.get("data", {}).get("question") if content_data else None



# LangChain tool setup
api_client = LeetCodeAPI()


def format_content(content):
    """Convert HTML content to formatted markdown-like text"""
    soup = BeautifulSoup(content, 'html.parser')
    
    # Replace code tags with backticks
    for code in soup.find_all('code'):
        code.replace_with(f'`{code.get_text()}`')
    
    # Replace strong tags with bold markers
    for strong in soup.find_all('strong'):
        strong.replace_with(f'**{strong.get_text()}**')
    
    # Format pre blocks as code examples
    for pre in soup.find_all('pre'):
        pre_content = pre.get_text().strip()
        pre.replace_with(f'\n```\n{pre_content}\n```\n')
    
    # Handle paragraphs and line breaks
    for p in soup.find_all('p'):
        p.append('\n\n')
        p.unwrap()
    
    # Clean up HTML entities and whitespace
    text = soup.get_text()
    text = text.replace('\xa0', ' ')  # Replace non-breaking spaces
    text = '\n'.join(line.strip() for line in text.split('\n'))
    text = '\n'.join(filter(None, text.split('\n')))  # Remove empty lines
    
    return text

def dict_to_string(data):
    """Convert problem dictionary to structured string"""
    sections = []
    
    # Basic Information
    sections.append(f"Question ID: {data['questionId']}")
    sections.append(f"Title: {data['title']}")
    sections.append(f"Difficulty: {data['difficulty']}\n")
    
    # Formatted Content
    formatted_content = format_content(data['content'])
    sections.append("Problem Description:\n" + formatted_content + "\n")
    
    # Topic Tags
    tags = [tag['name'] for tag in data['topicTags']]
    sections.append(f"Related Topics: {', '.join(tags)}")
    
    return '\n'.join(sections)


if __name__ == "__main__":

    print(dict_to_string(api_client.get_problem_data("140")))