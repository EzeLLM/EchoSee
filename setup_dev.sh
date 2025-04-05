# Load LangSmith API key from environment or .env file
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Set LangSmith tracing
export LANGSMITH_TRACING=true

# Verify LangSmith API key is set
if [ -z "$LANGSMITH_API_KEY" ]; then
    echo "Warning: LANGSMITH_API_KEY is not set. Please set it in your environment or .env file."
fi

