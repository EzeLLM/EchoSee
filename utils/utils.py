# utils/utils.py

import yaml
import inspect
import os
import dotenv
import atexit
from langchain_openai import ChatOpenAI
from smolagents import LiteLLMModel

# Import from local modules using explicit relative or absolute imports
from event_manager.event_manager import EventManager  # Assuming proper package structure

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
    event_manager_instance.stop()
    print("Event manager stopped.")

# Configuration loading
tts_config = open_yaml(CONFIG_PATH, 'TTS')
llm_config = open_yaml(CONFIG_PATH, 'LLM')

# Model initialization
litellm_llm = LiteLLMModel(
    "openai/gpt-4o-mini",
    api_key=os.environ['OPENAI_API_KEY']
)

llm = ChatOpenAI(
    api_key=os.environ['OPENAI_API_KEY'],
    model='gpt-4o-mini'
)

# Event manager initialization (renamed to avoid naming conflict)
event_manager_instance = EventManager()
event_manager_instance.start()

# Register cleanup
atexit.register(handle_exit)