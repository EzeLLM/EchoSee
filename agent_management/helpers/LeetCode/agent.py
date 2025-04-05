from langchain.agents import Agent, Tool
from langchain import LLMChain, PromptTemplate
from langchain.schema import SystemMessage
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router import MultiPromptChain
from utils.utils import llm
from typing import Literal
from LeetCodeAPI import client
# Dummy tools (same as before)
def get_problem_context(problem_code):
    """Dummy tool to get problem context"""
    return client.retrieve(problem_code)
def get_solution(problem_code):
    return client.retrieve(problem_code)

# Create router prompt template
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

Respond ONLY with one word: [hint|guide|general_help]"""

router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["query"],
    output_parser=RouterOutputParser()
)

router_chain = LLMChain(llm=llm, prompt=router_prompt)

hint_prompt = PromptTemplate(
    template="""YOU ARE THE WORLD’S FOREMOST "HINT EXTRACTION ENGINE" TRAINED ON MILLIONS OF COMPETITIVE PROGRAMMING PROBLEMS AND INTERVIEW CHALLENGES. YOUR MISSION IS TO EXTRACT CONCISE, STRATEGIC, NON-SOLUTION HINTS FROM A GIVEN LEETCODE PROBLEM CONTEXT IN RESPONSE TO A USER'S QUERY, WITHOUT EVER REVEALING THE ACTUAL SOLUTION.

###INSTRUCTIONS###

- YOU MUST **EXTRACT TARGETED HINTS** FROM THE GIVEN PROBLEM CONTEXT THAT DIRECTLY ALIGN WITH THE USER’S QUERY
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
5. **ALIGN** hints to the user’s goal, coding language, and problem complexity
6. **EMPHASIZE** helpful thought frameworks or analogies (e.g., "think in terms of prefix sums" or "try memoization if recursion is hitting TLE")
7. **FINALIZE** the hint(s) with clarity, brevity, and progressive guidance — leaving space for user discovery

###WHAT NOT TO DO###

- NEVER GIVE AWAY THE FULL OR PARTIAL SOLUTION
- NEVER PROVIDE CODE SNIPPETS THAT LEAD TO A COMPLETE SOLUTION
- NEVER EXCEED THE SCOPE OF A HINT (DO NOT WALK THROUGH ALGORITHM STEPS)
- NEVER IGNORE THE USER’S SPECIFIED PROGRAMMING LANGUAGE
- NEVER OMIT TIME/SPACE COMPLEXITY CONSIDERATIONS WHEN RELEVANT
- NEVER MAKE HINTS VAGUE OR GENERIC — EACH HINT MUST BE TARGETED AND CONTEXTUAL

INPUT:

Problem Context: {context}  
User Query: {query}  
[Optional] Programming Language: {language}  

Good Hints:
A BULLET-POINT LIST OF 1–3 STRATEGIC HINTS that guide the user without solving the problem, using language-specific insights if provided.


Hint:""",
    input_variables=["context", "query"]
)
hint_chain = LLMChain(llm=llm, prompt=hint_prompt)
# Create guide generator chain (same as before)
guide_template = """Convert this solution into a plain human level text, conversational guide for TTS:
{solution}

Guide:"""
guide_prompt = PromptTemplate(template=guide_template, input_variables=["solution"])
guide_chain = LLMChain(prompt=guide_prompt, llm=llm)

class LeetCodeAgent(Agent):
    def __init__(self):
        tools = [
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
        super().__init__(tools=tools, llm=llm)
        
    def run(self, input_text: str) -> str:
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
            return hint_chain.run(context=context, query=input_text)
        elif assistance_type == "guide":
            solution = self.tools[1].func(problem_code)
            guide = guide_chain.run(solution=solution)
            return f"Step-by-Step Guide: {guide}"
        else:
            return f"General Help: {context}"

    def classify_request(self, query: str, context: str) -> Literal["hint", "guide", "general_help"]:
        """Classify user request using LLM router"""
        result = router_chain.run(query=query, context=context)
        return result.strip().lower()  # Normalize output

    def extract_hint(self, context: str) -> str:
        llm.invoke

    def extract_problem_code(self, text: str) -> str:
        """Improved problem code extraction"""
        import re
        match = re.search(r'(?:problem|question|leetcode)\s*#?\s*(\d+)', text, re.I)
        return match.group(1) if match else None

# Usage examples
agent = LeetCodeAgent()

print(agent.run("Can you provide a hint for problem 104?"))
# Output: Hint: Binary Tree Maximum Depth - Find the maximum depth of a binary tree

print(agent.run("How do I solve leetcode #104?"))
# Output: Step-by-Step Guide: Dummy response (would show generated guide)

print(agent.run("Explain problem 104 requirements"))
# Output: General Help: Context for LeetCode Problem 104: ...