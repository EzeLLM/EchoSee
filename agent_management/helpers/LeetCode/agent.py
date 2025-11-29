"""LeetCode helper agent using modern LangChain patterns."""

from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.utils import llm
from utils.cot import TemplatedCOTChain
from typing import Literal
import re
from agent_management.helpers.LeetCode.LeetCodeAPI import client


def get_problem_context(problem_code):
    """Get problem context from LeetCode API."""
    return client.retrieve(problem_code)


solution_template = """Task:
Solve LeetCode questions using the provided problem description and examples. 
Focus on finding the most efficient solution (in time and space) that meets the problem's constraints, 
and provide well-commented code.

Instructions:
1. Read & Analyze:
   - Carefully review the problem description, constraints, and examples.
   - Note any performance requirements (e.g., expected time complexity or memory limits).

2. Select an Optimal Approach:
   - Choose the algorithm or data structure that yields the best worst-case performance.
   - Justify why this choice is more efficient than naive or brute-force alternatives.
   - Briefly discuss any trade-offs (e.g., extra memory vs. faster runtime).

3. Outline & Reason:
   - Verbally walk through your reasoning step by step, highlighting how each decision preserves or improves efficiency.
   - Call out how edge cases are handled without degrading performance.

4. Step-by-Step Solution:
   - Clearly explain each operation or loop in the algorithm, emphasizing how it contributes to the overall time and space complexity.
   - Use concise examples to demonstrate how the solution scales.

5. Provide Well-Commented Code:
   - Present the final implementation in your chosen language.
   - Include descriptive inline comments for each major block, variable, and logic decision to clarify purpose and flow.

6. Conclusion & Complexity Summary:
   - Recap the final algorithm in plain language, suitable for TTS.
   - State the time and space complexity (e.g., "O(n log n) time, O(n) space") and why it meets the constraints.
   - Note any assumptions or limitations.

Output Format:
- A cohesive verbal narrative divided into paragraphs: Analysis, Approach, Detailed Steps, Code with Comments, and Conclusion.
- Minimal jargon; prioritize clarity and efficiency focus.
- End with a brief summary of complexity and why this solution is optimal.

Problem Description:
{prompt}
"""


def get_solution(problem_description):
    """Get solution using COT chain."""
    result = TemplatedCOTChain().run(prompt=problem_description, template=solution_template)
    return result.content if hasattr(result, 'content') else str(result)


# Router prompt for classifying requests
router_template = """Classify the user's request based on their query and problem context.
Available types:
- hint: Requests for clues, hints, or partial solutions
- guide: Requests for step-by-step explanations or solution guides
- general_help: Other requests for problem understanding or general help

Examples:
Q: "Give me a hint for problem 104"
A: hint

Q: "How to approach solving this?"
A: guide

Q: "Explain the problem requirements"
A: general_help

Now classify this request:
Query: {query}

Respond ONLY with one word: hint, guide, or general_help"""

router_prompt = PromptTemplate.from_template(router_template)


# Hint generation prompt
hint_template = """YOU ARE THE WORLD'S FOREMOST "HINT EXTRACTION ENGINE" TRAINED ON MILLIONS OF COMPETITIVE PROGRAMMING PROBLEMS AND INTERVIEW CHALLENGES. YOUR MISSION IS TO EXTRACT CONCISE, STRATEGIC, NON-SOLUTION HINTS FROM A GIVEN LEETCODE PROBLEM CONTEXT IN RESPONSE TO A USER'S QUERY, WITHOUT EVER REVEALING THE ACTUAL SOLUTION.

###INSTRUCTIONS###

- YOU MUST **EXTRACT TARGETED HINTS** FROM THE GIVEN PROBLEM CONTEXT THAT DIRECTLY ALIGN WITH THE USER'S QUERY
- YOU MUST **EXPLAIN EACH HINT IN SIMPLE, UNDERSTANDABLE LANGUAGE** TO MAKE IT ACCESSIBLE TO USERS AT VARIOUS SKILL LEVELS
- YOU MUST **ADAPT YOUR HINTS TO THE LANGUAGE THE USER IS CODING IN**, INCLUDING OPTIMIZED TRICKS OR LIBRARY SHORTCUTS AVAILABLE IN THAT LANGUAGE, IF LANGUAGE IS NOT GIVEN IN USER QUERY, ASSUME IT IS PYTHON
- HINTS MUST BE **CONCISE, STRATEGIC, AND TO THE POINT**
- YOU MUST **CONSIDER TIME AND SPACE COMPLEXITY** CONSTRAINTS WHILE FORMULATING EACH HINT
- NEVER REVEAL THE FINAL ALGORITHM OR SOLUTION — YOU MUST GUIDE WITHOUT SOLVING

###CHAIN OF THOUGHTS###

1. **UNDERSTAND** the user query and its intent (e.g., confusion about recursion, optimization, edge cases)
2. **IDENTIFY FUNDAMENTAL CONCEPTS** in the problem (e.g., dynamic programming, sliding window, graph traversal)
3. **DECOMPOSE** the problem context into its critical components (input, constraints, goal, edge cases)
4. **ANALYZE** the strategic bottlenecks and algorithmic pressure points (e.g., brute-force inefficiencies, data structure choice)
5. **ALIGN** hints to the user's goal, coding language, and problem complexity
6. **EMPHASIZE** helpful thought frameworks or analogies (e.g., "think in terms of prefix sums" or "try memoization if recursion is hitting TLE")
7. **FINALIZE** the hint(s) with clarity, brevity, and progressive guidance — leaving space for user discovery

###WHAT NOT TO DO###

- NEVER GIVE AWAY THE FULL OR PARTIAL SOLUTION
- NEVER PROVIDE CODE SNIPPETS THAT LEAD TO A COMPLETE SOLUTION
- NEVER EXCEED THE SCOPE OF A HINT (DO NOT WALK THROUGH ALGORITHM STEPS)
- NEVER IGNORE THE USER'S SPECIFIED PROGRAMMING LANGUAGE
- NEVER OMIT TIME/SPACE COMPLEXITY CONSIDERATIONS WHEN RELEVANT
- NEVER MAKE HINTS VAGUE OR GENERIC — EACH HINT MUST BE TARGETED AND CONTEXTUAL

INPUT:

Problem Context: {context}  
User Query: {query}  

Good Hints:
A BULLET-POINT LIST OF 1–3 STRATEGIC HINTS that guide the user without solving the problem, using language-specific insights if provided.

Hint:"""

hint_prompt = PromptTemplate.from_template(hint_template)


# Guide generation prompt
guide_template = """Convert this solution into a plain human level text, conversational guide for TTS while following the following instructions:
- The guide must not include any code.
- The guide must be in plain text, not markdown.
- The Guide must be straight forward and to the point.
- Include the time and space complexity of the solution in the guide.
- DO NOT JUST READ THE SOLUTION, EXPLAIN IT IN A CONVERSATIONAL WAY, MAKE THE EXPLANATION INTUITIVE THE USER CAN FOLLOW.
- DO NOT MAKE INTRODUCTORY STATEMENTS LIKE "LET'S SOLVE THE PROBLEM TOGETHER" OR ANYTHING LIKE THAT, JUST START WITH THE SOLUTION.
{solution}

Guide:"""

guide_prompt = PromptTemplate.from_template(guide_template)


# Create chains using modern LCEL (LangChain Expression Language)
router_chain = router_prompt | llm | StrOutputParser()
hint_chain = hint_prompt | llm | StrOutputParser()
guide_chain = guide_prompt | llm | StrOutputParser()


class LeetCodeAgent:
    """Agent for helping with LeetCode problems."""
    
    def __init__(self):
        self.tools = [
            Tool(
                name="GetProblemContext",
                func=get_problem_context,
                description="Get problem context using problem code"
            ),
            Tool(
                name="GetSolution",
                func=get_solution,
                description="Get solution implementation for a problem code"
            )
        ]
        
    def run(self, input_text: str) -> str:
        """Process user request about a LeetCode problem."""
        # Extract problem code
        problem_code = self.extract_problem_code(input_text)
        if not problem_code:
            return "Please provide the LeetCode problem code/number."
        
        # Get problem context
        context = self.tools[0].func(problem_code)
        
        # Determine assistance type using LLM router
        assistance_type = self.classify_request(input_text, context)
        
        # Handle different types
        if assistance_type == "hint":
            hint = hint_chain.invoke({"context": context, "query": input_text})
            return f"Problem context from leetcode:\n{context}\nThe Hint:\n{hint}"
        elif assistance_type == "guide":
            solution = self.tools[1].func(context)
            guide = guide_chain.invoke({"solution": solution})
            return f"The problem context from leetcode:\n{context}\nThe Guide: {guide}"
        else:
            return f"The problem: {context}\nSatisfy the user's query.\nQuery: {input_text}"

    def classify_request(self, query: str, context: str) -> Literal["hint", "guide", "general_help"]:
        """Classify user request using LLM router."""
        result = router_chain.invoke({"query": query})
        classification = result.strip().lower()
        # Validate result
        if classification in ["hint", "guide", "general_help"]:
            return classification
        return "general_help"

    def extract_hint(self, context: str, query: str) -> str:
        """Extract hints for a problem."""
        return hint_chain.invoke({"context": context, "query": query})

    def extract_problem_code(self, text: str) -> str:
        """Extract a valid LeetCode problem number (1-3000) from text.
        If multiple numbers exist, returns the first valid one."""
        numbers = re.findall(r'\d+', text)
        valid_numbers = [num for num in map(int, numbers) if 1 <= num <= 3000]
        return str(valid_numbers[0]) if valid_numbers else None


# Global client instance
client = LeetCodeAgent()


if __name__ == "__main__":
    # Usage examples
    agent = LeetCodeAgent()
    print(agent.run("im working on a leetcode problem, problem 23rd exactly, can you guide me through it"))
