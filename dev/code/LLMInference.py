import os
from groq import Groq
import FuncHub
from dotenv import load_dotenv
load_dotenv()
GROQAPI = os.getenv('GROQAPI')

class LLMInference:
    def __init__(self):
        self.client = Groq(
            api_key=GROQAPI,
        )

    def message_appender(self, history, role, content):
        new_message = {
            "role": role,
            "content": content
        }
        history.append(new_message)
        return history

    def infer_groq(self, messages, model = "llama-3.1-70b-versatile",max_tokens=256, temperature=0.7): 
        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return chat_completion.choices[0].message.content

    def load_history(self, history_path):
        history = FuncHub.open_json(history_path)
        return history
    
    def llm(self, message: str):
        messages = self.load_history("history/unrestricted_fun.json")
        messages = self.message_appender(messages, "user", message)
        result = self.infer_groq(messages)
        messages = self.message_appender(messages, "assistant", result)
        return result
    
if __name__ == "__main__":
   llm_ = LLMInference()
   while True:
       user_input = input("You: ")
       print("\nModel: ",llm_.llm(user_input))

