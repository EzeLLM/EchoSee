import google.generativeai as genai
import os
import dotenv
import inspect

dotenv.load_dotenv()
genai.configure(api_key=os.getenv("GEMINIAPI"))

NON_TOOL_METHODS = ['__init__', 'get_tool_list', 'get_function_info', 'get_available_tools', 'process_tool_calls']

def get_tool_list():
    """
    Retrieve all function objects from the global scope of this module,
    excluding the ones listed in NON_TOOL_METHODS.
    
    Returns:
        list: List of function objects
    """
    functions = []
    for name, obj in globals().items():
        if inspect.isfunction(obj) and name not in NON_TOOL_METHODS:
            functions.append(obj)
    return functions

def get_function_info(func):
    """
    Retrieve the docstring of the given function.
    
    Args:
        func (function): The function object
    
    Returns:
        str: The docstring of the function
    """
    return func.__doc__

def get_available_tools():
    """
    Get the available tools (functions) from the global scope.
    
    Returns:
        list: The list of available tools (function objects)
    """
    return get_tool_list()

def calculate(expression: str) -> float:
    """
    Calculate a mathematical expression

    Args:
        expression (str): The expression to be calculated
    
    Returns:
        float: The result of the expression
    """
    return eval(expression)

def power_disco_ball(power: bool) -> bool:
    """
    Powers the spinning disco ball.

    Args:
        power (bool): Whether to power the disco ball or not

    Returns:
        bool: Whether the disco ball is spinning or not
    """
    print(f"Disco ball is {'spinning!' if power else 'stopped.'}")
    return True

def process_tool_calls(response):
    """
    Process tool function calls and return their responses.
    
    Args:
        response: The response containing function call parts
    
    Returns:
        dict: A dictionary with function names and arguments
    """
    responses = {}
    for part in response.parts:
        if fn := part.function_call:
            args = {key: val for key, val in fn.args.items()}
            responses[fn.name] = args
    return responses

available_tools = get_available_tools()

func_to_name_dict = {func: func.__name__ for func in available_tools}


