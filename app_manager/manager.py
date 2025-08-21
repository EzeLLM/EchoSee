from agent_management import agent_manager
from utils.utils import llm, llm_config
from tts.tts import TTS
from agent_management.agent_manager import AgentManager
from langchain_core.messages import HumanMessage
from utils.utils import *
from stt.stt import STT
from logger import logger as _logger

log = _logger.Logger("app_manager")
def main():
    # Initialize TTS
    tts = TTS( )
    print("Voice Assistant started! Press Enter after typing your question (type 'quit' to exit)")
    log.info("App manager initialized; voice assistant ready.")
    am = AgentManager()
    stt = STT()
    while True:
        try:
            # Get user input
            # user_query = stt.listen_and_transcribe_key()
            user_query = input("You: ")
            log.debug(f"Received user input: {user_query}")
            # Check for quit command
            if user_query.lower() == 'quit':
                print("Goodbye!")
                break
                
            # Process query through agent
            log.info("Dispatching message to AgentManager.process_message")
            response = am.process_message(user_query)
            
            # Extract the assistant's response
            assistant_message = response[-1].content
            log.debug(f"Assistant raw response messages count: {len(response)}")
            log.info("Assistant response extracted; printing to console")
            print(f"\nAssistant: {assistant_message}")
            
            # Convert response to speech
            # tts.play_with_device(assistant_message, device=tts_config['device'])
            
        except Exception as e:
            log.error(f"Unhandled error in main loop: {e}")
            print(f"An error occurred: {e}")
            continue

if __name__ == "__main__":
    main()
