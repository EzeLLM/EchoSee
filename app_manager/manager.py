import sys
import json


def health_command():
    """Run health check and print results."""
    from core.health_check import health_checker
    
    result = health_checker.check_all()
    print(json.dumps(result, indent=2))
    
    # Return exit code based on health status
    return 0 if result['status'] == 'healthy' else 1


def main():
    # Import heavy modules only when running main app
    from agent_management import agent_manager
    from core.config_manager import config
    from core.app_context import app_context
    from tts.tts import TTS
    from agent_management.agent_manager import AgentManager
    from langchain_core.messages import HumanMessage
    from stt.stt import STT
    # Initialize application context (must be first!)
    app_context.initialize()

    # Get TTS config
    tts_config = config.get_section('TTS')

    # Initialize TTS
    tts = TTS()
    print("Voice Assistant started! Press Enter after typing your question (type 'quit' to exit)")
    am = AgentManager()
    stt = STT()
    while True:
        try:
            # Get user input
            user_query = stt.listen_and_transcribe_key()
            # user_query = input("You: ")
            # Check for quit command
            if user_query.lower() == 'quit':
                print("Goodbye!")
                break
                
            # Process query through agent
            response = am.process_message(user_query)
            
            # Extract the assistant's response
            assistant_message = response[-1].content
            print(f"\nAssistant: {assistant_message}")
            
            # Convert response to speech
            tts.play_with_device(assistant_message, device=tts_config['device'])
            
        except Exception as e:
            print(f"An error occurred: {e}")
            continue

if __name__ == "__main__":
    # Handle CLI commands
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "health":
            exit_code = health_command()
            sys.exit(exit_code)
        elif command == "help":
            print("EchoSee Voice Assistant")
            print()
            print("Usage: python -m app_manager.manager [command]")
            print()
            print("Commands:")
            print("  (none)    Start the voice assistant")
            print("  health    Run health check and show component status")
            print("  help      Show this help message")
            sys.exit(0)
        else:
            print(f"Unknown command: {command}")
            print("Run 'python -m app_manager.manager help' for usage")
            sys.exit(1)
    
    # Default: run main application
    main()
