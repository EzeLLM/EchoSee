from typing import Optional, Dict, Any
import requests
import json
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os

class LeetCodeAPI:
    def __init__(self):
        load_dotenv()
        self.graphql_url = "https://leetcode.com/graphql"
        self.session = requests.Session()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
            "Content-Type": "application/json",
            "Referer": "https://leetcode.com",
            "Origin": "https://leetcode.com"
        }
        # Mapping of common problem IDs to their title slugs
        self.common_problems = {
            "1": "two-sum",
            "2": "add-two-numbers",
            # Add more mappings as needed
            "140": "word-break-ii"
        }

    def _graphql_query(self, query: str, variables: Dict[str, Any]) -> Optional[Dict]:
        """Execute GraphQL query with error handling"""
        try:
            response = self.session.post(
                self.graphql_url,
                json={"query": query, "variables": variables},
                headers=self.headers,
                timeout=10
            )
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API Request failed: {str(e)}")
            return None
        except json.JSONDecodeError:
            print("Invalid JSON response")
            return None

    def _get_title_slug(self, problem_code: str) -> str:
        """Get the problem title slug from its code/ID"""
        try:
            # First try to get the problem by ID
            response = requests.get("https://leetcode.com/api/problems/all/")
            if response.status_code == 200:
                problems_data = response.json()
                for problem in problems_data.get("stat_status_pairs", []):
                    if str(problem.get("stat", {}).get("question_id")) == problem_code:
                        return problem.get("stat", {}).get("question__title_slug")
            
            # If not found by ID, check the common problems mapping
            if problem_code in self.common_problems:
                return self.common_problems[problem_code]
            
            # Last resort: use the problem code as slug directly
            return problem_code
            
        except Exception as e:
            print(f"Error getting title slug: {e}")
            # Fallback to using the mapping or the code itself
            return self.common_problems.get(problem_code, problem_code)

    def _format_content(self, content: str) -> str:
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

    def _dict_to_string(self, data: Dict) -> str:
        """Convert problem dictionary to structured string"""
        sections = []
        
        # Basic Information
        sections.append(f"Question ID: {data['questionId']}")
        sections.append(f"Title: {data['title']}")
        sections.append(f"Difficulty: {data['difficulty']}\n")
        
        # Formatted Content
        formatted_content = self._format_content(data['content'])
        sections.append("Problem Description:\n" + formatted_content + "\n")
        
        # Topic Tags
        tags = [tag['name'] for tag in data['topicTags']]
        sections.append(f"Related Topics: {', '.join(tags)}")
        
        return '\n'.join(sections)

    def retrieve(self, problem_code: str) -> Optional[str]:
        """Retrieve and format problem data from LeetCode"""
        title_slug = self._get_title_slug(problem_code)
        
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
        
        content_vars = {"titleSlug": title_slug}
        content_data = self._graphql_query(content_query, content_vars)
        
        if content_data and "data" in content_data and "question" in content_data["data"]:
            return self._dict_to_string(content_data["data"]["question"])
        else:
            return f"Failed to retrieve data for problem code: {problem_code}"


# Example usage
client = LeetCodeAPI()
if __name__ == "__main__":
    problem_info = client.retrieve("141")
    print(problem_info)