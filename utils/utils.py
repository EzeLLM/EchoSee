# utils/utils.py

import yaml
import inspect
import os
import dotenv
import atexit
from langchain_openai import ChatOpenAI
from smolagents import LiteLLMModel
from langchain_deepseek import ChatDeepSeek

# Load environment variables
dotenv.load_dotenv()

CONFIG_PATH = 'config.yml'

def open_yaml(file_path, key=None):
    """Safe YAML file loader with optional key filtering"""
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
            return data.get(key) if key else data
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing {file_path}: {e}")
        return None
    except AttributeError:
        print(f"Key {key} not found in file {file_path}.")
        return None

def handle_exit():
    """Cleanup function for exit handling"""
    print("Exiting...")

# Configuration loading
tts_config = open_yaml(CONFIG_PATH, 'TTS')
llm_config = open_yaml(CONFIG_PATH, 'LLM')

# Model initialization
litellm_llm = LiteLLMModel(
    f"{llm_config['provider']}/{llm_config['model']}",
    api_key=os.environ['OPENAI_API_KEY']
)

if llm_config['provider'] == 'openai':
    llm = ChatOpenAI(
        api_key=os.environ['OPENAI_API_KEY'],
        model=f"{llm_config['model']}"
    )
else:
    raise ValueError(f"Invalid provider: {llm_config['provider']}")



if llm_config['high_performance_provider'] == 'deepseek':
    high_performance_llm = ChatDeepSeek(
        api_key=os.environ['DEEPSEEK_API_KEY'],
        model=f"{llm_config['high_performance_model']}"
    )
elif llm_config['high_performance_provider'] == 'openai':
    high_performance_llm = ChatOpenAI(
        api_key=os.environ['OPENAI_API_KEY'],
        model=f"{llm_config['high_performance_model']}"
    )
else:
    raise ValueError(f"Invalid provider: {llm_config['high_performance_provider']}")

# Register cleanup
atexit.register(handle_exit)