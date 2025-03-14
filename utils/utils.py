import yaml
import inspect
from langchain_openai import ChatOpenAI
import os
import dotenv
from smolagents import LiteLLMModel
dotenv.load_dotenv()

def open_yaml(file_path, key=None):
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
            if key is not None:
                return data.get(key)
            else:
                return data
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing {file_path}: {e}")
        return None
    except AttributeError:
        print(f"Key {key} not found in file {file_path}.")
        return None


litellm_llm = LiteLLMModel("openai/gpt-4o-mini",api_key=os.environ['OPENAI_API_KEY'])
llm = ChatOpenAI(api_key=os.environ['OPENAI_API_KEY'], model='gpt-4o-mini')
llm_config = open_yaml("config.yml", "LLM")