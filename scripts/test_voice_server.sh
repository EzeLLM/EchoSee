#!/bin/bash

# Test script for Voice Server
# Checks all prerequisites and dependencies

echo "=== EchoSee Voice Server Test ==="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track overall status
ALL_PASSED=true

# Test 1: Check Python
echo -n "Checking Python... "
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1)
    echo -e "${GREEN}✓${NC} $PYTHON_VERSION"
else
    echo -e "${RED}✗${NC} Python not found"
    ALL_PASSED=false
fi

# Test 2: Check FFmpeg
echo -n "Checking FFmpeg... "
if command -v ffmpeg &> /dev/null; then
    FFMPEG_VERSION=$(ffmpeg -version 2>&1 | head -1)
    echo -e "${GREEN}✓${NC} Found"
else
    echo -e "${RED}✗${NC} FFmpeg not found. Install with: sudo apt-get install ffmpeg"
    ALL_PASSED=false
fi

# Test 3: Check Python packages
echo -n "Checking Flask... "
if python -c "import flask" 2>/dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC} Flask not installed"
    ALL_PASSED=false
fi

echo -n "Checking flask-cors... "
if python -c "import flask_cors" 2>/dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC} flask-cors not installed"
    ALL_PASSED=false
fi

echo -n "Checking LangGraph... "
if python -c "import langgraph" 2>/dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC} LangGraph not installed. Run: pip install -e ."
    ALL_PASSED=false
fi

echo -n "Checking EchoSee modules... "
if python -c "from agent_management.agent_manager import AgentManager" 2>/dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC} EchoSee not installed. Run: pip install -e ."
    ALL_PASSED=false
fi

# Test 4: Check environment variables
echo -n "Checking .env file... "
if [ -f ".env" ]; then
    echo -e "${GREEN}✓${NC}"

    echo -n "Checking OPENAI_API_KEY... "
    if grep -q "OPENAI_API_KEY=" .env; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${YELLOW}⚠${NC} Not found in .env"
    fi
else
    echo -e "${RED}✗${NC} .env file not found"
    ALL_PASSED=false
fi

# Test 5: Check port availability
echo -n "Checking port 9004... "
if lsof -Pi :9004 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠${NC} Port already in use"
else
    echo -e "${GREEN}✓${NC} Available"
fi

echo ""
if [ "$ALL_PASSED" = true ]; then
    echo -e "${GREEN}All checks passed!${NC} Ready to run the voice server."
    echo ""
    echo "To start:"
    echo "  ./scripts/launch_voice_ui.sh"
    echo ""
    echo "Or manually:"
    echo "  python -m api.voice_server"
else
    echo -e "${RED}Some checks failed.${NC} Please fix the issues above."
    exit 1
fi
