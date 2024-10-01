import os
from groq import Groq
import FuncHub
from dotenv import load_dotenv
import LLMTools.LLMTools_Groq as groq_tools
import LLMTools.LLMTools_GEMINI as gemini_tools
import google.generativeai as genai
import json
load_dotenv()
GROQAPI = os.getenv('GROQAPI')
genai.configure(api_key=os.getenv("GEMINIAPI"))

class LLMInference:
    def __init__(self,config):
        self.client = Groq(
            api_key=GROQAPI,
        )
        self.config = FuncHub.open_yaml(config,'LLMInference')
        self.history_path = self.config['history_path']
        self.tools:bool = self.config['tools']
        self.llm_host = self.config['llm_host']
        self.llm_model = self.config['llm_model']

        if self.llm_host == "groq":
            if self.tools:
                self.llm_tools = groq_tools.LLMTools()
        elif self.llm_host == "gemini":
            if self.tools:
                self.llm_tools = gemini_tools
                # print(self.llm_tools.get_available_tools())
                self.model = genai.GenerativeModel(model_name=self.llm_model, tools=self.llm_tools.available_tools)
            else:
                self.model = genai.GenerativeModel(model_name=self.llm_model)
            self.chat = self.model.start_chat()
            # print((self.chat.send_message("Hello!")).text)

        self.messages = self.load_history(self.history_path)

    def message_appender(self, history:list, role:str, content:str):
        new_message = {
            "role": role,
            "content": content
        }
        history.append(new_message)

        return history

    def infer_groq(self, messages, model ,max_tokens=256, temperature=0.7): 
        
        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return chat_completion.choices[0].message.content
    
    def infer_groq_with_tools(self, messages, model, tool_choice="auto", max_tokens=256, temperature=0.7):
        tools = self.llm_tools.get_tools()
        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model=model,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if chat_completion.choices[0].message.tool_calls:
            messages = self.llm_tools.proccess_tool_calls(chat_completion.choices[0].message.tool_calls, messages)
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        return chat_completion.choices[0].message.content
    
    def infer_gemini(self, messages, max_tokens=256, temperature=0.7):
        response = self.chat.send_message(messages)
        return response.text
    
    def infer_gemini_with_tools(self, message,max_tokens=256, temperature=0.7):
        response = self.chat.send_message(message)
        responses = self.llm_tools.process_tool_calls(response)
        if responses:
            response_parts = [
            genai.protos.Part(function_response=genai.protos.FunctionResponse(name=fn, response={"result": val}))
            for fn, val in responses.items()
            ]
            response = self.chat.send_message(response_parts)
        return response.text

    def load_history(self, history_path):
        history = FuncHub.open_json(history_path)
        return history
    
    def llm(self, message: str):
        if self.llm_host == "groq":
            self.messages = self.message_appender(self.messages, "user", message)

            if self.tools:
                result = self.infer_groq_with_tools(messages=self.messages, model=self.llm_model)
            else:
                result = self.infer_groq(messages=self.messages, model=self.llm_model)
            
            self.messages = self.message_appender(self.messages, "assistant", result)

        elif self.llm_host == "gemini":
            if self.tools:
                result = self.infer_gemini_with_tools(message)
            else:
                result = self.infer_gemini(message)
        else:
            raise Exception("LLM Host not recognized")
        return result
    
if __name__ == "__main__":
   llm_ = LLMInference('dev/code/config/echosee.yaml')
   while True:
       user_input = input("You: ")
       print("\nModel: ",llm_.llm(user_input),"\n")

