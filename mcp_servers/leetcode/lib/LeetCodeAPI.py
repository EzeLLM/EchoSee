"""LeetCode API wrapper for problem retrieval."""

import requests
from typing import Dict, Any


class LeetCodeAPI:
    """Wrapper for LeetCode GraphQL API."""
    
    def __init__(self):
        self.base_url = "https://leetcode.com/graphql"
        self.headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0"
        }
    
    def retrieve(self, slug: str) -> Dict[str, Any]:
        """Retrieve problem details by slug.
        
        Args:
            slug: Problem slug (e.g., 'two-sum')
            
        Returns:
            Problem details including title, difficulty, description, etc.
        """
        query = """
        query getQuestionDetail($titleSlug: String!) {
            question(titleSlug: $titleSlug) {
                questionId
                title
                titleSlug
                content
                difficulty
                likes
                dislikes
                topicTags {
                    name
                    slug
                }
                codeSnippets {
                    lang
                    langSlug
                    code
                }
                sampleTestCase
                hints
            }
        }
        """
        
        variables = {"titleSlug": slug}
        
        try:
            response = requests.post(
                self.base_url,
                json={"query": query, "variables": variables},
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            question = data.get("data", {}).get("question", {})
            
            if not question:
                raise ValueError(f"Problem '{slug}' not found")
            
            return {
                "id": question.get("questionId"),
                "title": question.get("title"),
                "slug": question.get("titleSlug"),
                "difficulty": question.get("difficulty"),
                "content": question.get("content", ""),
                "topics": [tag["name"] for tag in question.get("topicTags", [])],
                "code_snippets": question.get("codeSnippets", []),
                "sample_test": question.get("sampleTestCase", ""),
                "hints": question.get("hints", [])
            }
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to retrieve problem: {str(e)}")
        except (KeyError, ValueError) as e:
            raise Exception(f"Failed to parse response: {str(e)}")
