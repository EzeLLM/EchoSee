import json
import inspect
import datetime
import requests

NON_TOOL_METHODS = ['__init__', 'get_function_info', 'call_by_refrence','proccess_tool_calls','get_tools','get_method','get_tool_dict']

class LLMTools:
    def __init__(self) -> None:
        self.available_tools = {
            method_[0]['function']['name']: method_[1] for method_ in self.get_tool_dict()
        }

    def get_function_info(self, func):
        func_doc = func.__doc__
        func_info = json.loads(func_doc)
        json_info = {
            "type": "function",
            "function": func_info
        }
        return json_info
    
    def get_method(self, method_name):
        return getattr(self, method_name)
    
    def get_tools(self):
        tools = []
        for name,tool in self.available_tools.items():
            tools.append(self.get_function_info(tool))
        return tools
    
    def get_tool_dict(self, obj=None):
        tools = []
        for name, obj in inspect.getmembers(self if obj is None else obj, inspect.ismethod):  
            if name not in NON_TOOL_METHODS:
                tools.append([self.get_function_info(obj), obj])
            
        return tools
    
    def call_by_refrence(self, obj, method_name, *args, **kwargs):

        method = getattr(obj, method_name)
        return method(*args, **kwargs)

    def calculate(self, expression): 
        '''{
        "name": "calculate",
        "description": "Evaluate a mathematical expression",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        }
        }'''

        try:
            result = eval(expression)
            return json.dumps({"result": result})
        except:
            return json.dumps({"error": "Invalid expression"})
        
    def get_date(self):
        '''{
        "name": "get_date",
        "description": "Get the current date",
        "parameters": {}
        }'''
        return json.dumps({"date": str(datetime.datetime.now())})
    
    def get_time(self):
        '''{
        "name": "get_time",
        "description": "Get the current time",
        "parameters": {}
        }'''
        return json.dumps({"time": str(datetime.datetime.now().time())})
    
    
    

    def proccess_tool_calls(self,tool_calls,messages):
        
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = self.available_tools[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(**function_args)
            # if function_name == 'calculate':
            #     function_response = function_to_call(expression=function_args.get('expression'))
            # else:
            #     function_response = function_to_call(**function_args)

            messages.append(
                {
                    "tool_call_id": tool_call.id, 
                    "role": "tool", 
                    "name": function_name,
                    "content": function_response,
                }
            )

        return messages

if __name__ == "__main__":
    llmtools = LLMTools()
    print(llmtools.get_tools())
    print(llmtools.available_tools)
