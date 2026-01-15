#!/bin/bash

# EchoSee Launch Script with Echonoti (WE - With Echonoti)
# =========================================================
# Complete launch script with environment setup, dependency checks,
# MCP server health checks, Echonoti PWA, and beautiful output.

set -e  # Exit on error

# ANSI Color Codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
DIM='\033[2m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Symbols
CHECK="✓"
CROSS="✗"
ARROW="→"
ROCKET="🚀"
GEAR="⚙"
WARNING="⚠"
INFO="ℹ"
PLUG="🔌"
GLOBE="🌐"

# Parse command-line arguments
SKIP_DEPS=false
SKIP_HEALTH=false
SKIP_ENV=false
SKIP_ALL=false

show_help() {
    echo -e "${CYAN}${BOLD}EchoSee Launch Script (With Echonoti)${NC}"
    echo -e "${DIM}Usage: ./launch-we.sh [OPTIONS]${NC}\n"
    echo -e "${BOLD}Options:${NC}"
    echo -e "  ${CYAN}--skip-deps${NC}      Skip dependency installation checks"
    echo -e "  ${CYAN}--skip-health${NC}    Skip MCP server health checks"
    echo -e "  ${CYAN}--skip-env${NC}       Skip environment variable validation"
    echo -e "  ${CYAN}--skip-all${NC}       Skip all checks (deps, health, env)"
    echo -e "  ${CYAN}-h, --help${NC}       Show this help message"
    echo ""
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-deps)
            SKIP_DEPS=true
            shift
            ;;
        --skip-health)
            SKIP_HEALTH=true
            shift
            ;;
        --skip-env)
            SKIP_ENV=true
            shift
            ;;
        --skip-all)
            SKIP_ALL=true
            SKIP_DEPS=true
            SKIP_HEALTH=true
            SKIP_ENV=true
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo -e "Use ${CYAN}--help${NC} for usage information"
            exit 1
            ;;
    esac
done

# Process IDs for cleanup
ECHONOTI_PID=""
ECHOSEE_PID=""

# Print functions
print_header() {
    clear
    echo -e "${CYAN}${BOLD}"
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║                                                                  ║"
    echo -e "║   ${MAGENTA}███████╗ ██████╗██╗  ██╗ ██████╗ ███████╗███████╗███████╗${CYAN}   ║"
    echo -e "║   ${MAGENTA}██╔════╝██╔════╝██║  ██║██╔═══██╗██╔════╝██╔════╝██╔════╝${CYAN}   ║"
    echo -e "║   ${MAGENTA}█████╗  ██║     ███████║██║   ██║███████╗█████╗  █████╗${CYAN}     ║"
    echo -e "║   ${MAGENTA}██╔══╝  ██║     ██╔══██║██║   ██║╚════██║██╔══╝  ██╔══╝${CYAN}     ║"
    echo -e "║   ${MAGENTA}███████╗╚██████╗██║  ██║╚██████╔╝███████║███████╗███████╗${CYAN}   ║"
    echo -e "║   ${MAGENTA}╚══════╝ ╚═════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚══════╝╚══════╝${CYAN}   ║"
    echo "║                                                                  ║"
    echo -e "║         ${DIM}Intelligent Voice Assistant Pipeline${CYAN}${BOLD}                  ║"
    echo -e "║                ${DIM}+ Echonoti PWA${CYAN}${BOLD}                                    ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_section() {
    echo -e "\n${BOLD}${CYAN}${1} ${2}${NC}"
    echo -e "${DIM}────────────────────────────────────────────────────────────────${NC}"
}

print_success() {
    echo -e "  ${GREEN}${CHECK}${NC} ${1}"
}

print_error() {
    echo -e "  ${RED}${CROSS}${NC} ${1}"
}

print_warning() {
    echo -e "  ${YELLOW}${WARNING}${NC} ${1}"
}

print_info() {
    echo -e "  ${BLUE}${INFO}${NC} ${1}"
}

print_step() {
    echo -e "  ${CYAN}${ARROW}${NC} ${DIM}${1}${NC}"
}

# Error handler
handle_error() {
    echo -e "\n${RED}${BOLD}${CROSS} Launch failed!${NC}"
    echo -e "${DIM}Error occurred in: ${1}${NC}\n"
    cleanup
    exit 1
}

# Cleanup function
cleanup() {
    echo -e "\n${DIM}════════════════════════════════════════════════════════════════${NC}"
    echo -e "\n${YELLOW}${INFO} Shutting down...${NC}"

    # Kill Echonoti if running
    if [ ! -z "$ECHONOTI_PID" ] && kill -0 $ECHONOTI_PID 2>/dev/null; then
        print_step "Stopping Echonoti PWA (PID: $ECHONOTI_PID)..."
        kill $ECHONOTI_PID 2>/dev/null || true
        wait $ECHONOTI_PID 2>/dev/null || true
        print_success "Echonoti stopped"
    fi

    # Kill EchoSee if running
    if [ ! -z "$ECHOSEE_PID" ] && kill -0 $ECHOSEE_PID 2>/dev/null; then
        print_step "Stopping EchoSee (PID: $ECHOSEE_PID)..."
        kill $ECHOSEE_PID 2>/dev/null || true
        wait $ECHOSEE_PID 2>/dev/null || true
        print_success "EchoSee stopped"
    fi

    echo -e "\n${CYAN}${INFO} All services stopped gracefully${NC}\n"
}

# Set up trap for cleanup on exit
trap cleanup EXIT INT TERM

# Main launch sequence
print_header

# ============================================================================
# 1. Environment Check
# ============================================================================
print_section "${GEAR}" "Environment Setup"

# Check if we're in the project directory
if [ ! -f "config.yml" ]; then
    print_error "Not in EchoSee project directory!"
    echo -e "  ${DIM}Please run this script from the project root.${NC}\n"
    exit 1
fi
print_success "Project directory confirmed"

# ============================================================================
# 2. Virtual Environment (Python)
# ============================================================================
print_step "Checking Python virtual environment..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_warning "Virtual environment not found, creating one..."
    python3 -m venv venv || handle_error "Virtual environment creation"
    print_success "Virtual environment created"
else
    print_success "Virtual environment found"
fi

# Activate virtual environment
print_step "Activating virtual environment..."
source venv/bin/activate || handle_error "Virtual environment activation"
print_success "Virtual environment activated"

# ============================================================================
# 3. Environment Variables
# ============================================================================
print_section "${INFO}" "Loading Environment Variables"

if [ -f ".env" ]; then
    print_step "Loading .env file..."
    export $(cat .env | grep -v '^#' | grep -v '^$' | xargs)
    print_success ".env file loaded"

    # Check required API keys (skip if requested)
    if [ "$SKIP_ENV" = false ]; then
        MISSING_KEYS=()

        if [ -z "$OPENAI_API_KEY" ]; then
            MISSING_KEYS+=("OPENAI_API_KEY")
        fi

        if [ -z "$TAVILY_API_KEY" ]; then
            MISSING_KEYS+=("TAVILY_API_KEY")
        fi

        if [ ${#MISSING_KEYS[@]} -gt 0 ]; then
            print_warning "Missing API keys: ${MISSING_KEYS[*]}"
            echo -e "  ${DIM}Some features may not work correctly.${NC}"
        else
            print_success "All required API keys present"
        fi

        # Optional keys
        if [ -n "$LANGSMITH_API_KEY" ]; then
            print_info "LangSmith tracing available"
        fi

        if [ -n "$DEEPSEEK_API_KEY" ]; then
            print_info "DeepSeek provider available"
        fi
    else
        print_info "Environment validation skipped (--skip-env)"
    fi
else
    print_error ".env file not found!"
    echo -e "  ${DIM}Create a .env file with required API keys:${NC}"
    echo -e "  ${CYAN}OPENAI_API_KEY=your_key${NC}"
    echo -e "  ${CYAN}TAVILY_API_KEY=your_key${NC}\n"
    exit 1
fi

# ============================================================================
# 4. Python Dependencies
# ============================================================================
print_section "${GEAR}" "Python Dependencies"

if [ "$SKIP_DEPS" = false ]; then
    print_step "Checking installed packages..."

    # Function to check if a package is installed
    check_package() {
        python -c "import $1" 2>/dev/null
        return $?
    }

    # Check critical packages
    CRITICAL_PACKAGES=("yaml:PyYAML" "openai:openai" "langchain_core:langchain_core" "langgraph:langgraph")
    MISSING_PACKAGES=()

    for pkg_spec in "${CRITICAL_PACKAGES[@]}"; do
        IFS=':' read -r import_name pkg_name <<< "$pkg_spec"
        if ! check_package "$import_name"; then
            MISSING_PACKAGES+=("$pkg_name")
        fi
    done

    if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
        print_warning "Missing packages detected, installing dependencies..."
        print_step "Running: pip install -r requirements.txt"
        pip install -q -r requirements.txt || handle_error "Package installation"
        print_success "Dependencies installed successfully"
    else
        print_success "All critical dependencies present"
    fi

    # Install in editable mode if not already done
    if ! python -c "import app_manager" 2>/dev/null; then
        print_step "Installing EchoSee in editable mode..."
        pip install -q -e . || handle_error "Editable installation"
        print_success "EchoSee package installed"
    fi
else
    print_info "Dependency checks skipped (--skip-deps)"
fi

# ============================================================================
# 5. Node.js and Echonoti Setup
# ============================================================================
print_section "${GLOBE}" "Echonoti PWA Setup"

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    print_error "Node.js is not installed!"
    echo -e "  ${DIM}Install Node.js from: https://nodejs.org/${NC}"
    echo -e "  ${DIM}Or use: brew install node${NC}\n"
    exit 1
fi

NODE_VERSION=$(node --version)
print_success "Node.js ${NODE_VERSION} found"

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    print_error "npm is not installed!"
    exit 1
fi

NPM_VERSION=$(npm --version)
print_success "npm ${NPM_VERSION} found"

# Check echonoti directory
if [ ! -d "echonoti" ]; then
    print_error "Echonoti directory not found!"
    echo -e "  ${DIM}Make sure echonoti/ exists in the project root.${NC}\n"
    exit 1
fi

# Check and install echonoti dependencies
cd echonoti

if [ ! -d "node_modules" ]; then
    print_warning "Echonoti dependencies not installed, installing..."
    print_step "Running: npm install"
    npm install || handle_error "Echonoti npm install"
    print_success "Echonoti dependencies installed"
else
    print_success "Echonoti dependencies found"
fi

cd ..

# ============================================================================
# 6. MCP Server Health Check
# ============================================================================
print_section "${PLUG}" "MCP Server Health Check"

if [ "$SKIP_HEALTH" = false ]; then
    print_step "Testing MCP server connections..."

    # Run MCP test script
    if python -m scripts.test_mcp --no-color > /tmp/echosee_mcp_test.log 2>&1; then
        # Parse the output for connected servers
        CONNECTED=$(grep -c "Connected" /tmp/echosee_mcp_test.log || echo "0")
        FAILED=$(grep -c "Failed" /tmp/echosee_mcp_test.log || echo "0")

        if [ "$CONNECTED" -gt 0 ]; then
            print_success "MCP servers operational (${CONNECTED} connected)"
            if [ "$FAILED" -gt 0 ]; then
                print_warning "${FAILED} server(s) failed to connect"
                echo -e "  ${DIM}Run 'python -m scripts.test_mcp' for details${NC}"
            fi
        else
            print_info "No MCP servers configured or connected"
        fi
    else
        # MCP might be disabled or no servers configured
        if grep -q "disabled" /tmp/echosee_mcp_test.log 2>/dev/null; then
            print_info "MCP support is disabled"
        elif grep -q "No MCP servers" /tmp/echosee_mcp_test.log 2>/dev/null; then
            print_info "No MCP servers configured"
        else
            print_warning "MCP health check had issues"
            echo -e "  ${DIM}Run 'python -m scripts.test_mcp' for details${NC}"
        fi
    fi

    # Cleanup
    rm -f /tmp/echosee_mcp_test.log
else
    print_info "MCP health checks skipped (--skip-health)"
fi

# ============================================================================
# 7. Configuration Summary
# ============================================================================
print_section "${INFO}" "Configuration Summary"

# Parse config.yml for current settings (disable exit on error for this section)
set +e
if command -v python &> /dev/null; then
    CONFIG_SUMMARY=$(python -c "
import yaml
with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

llm_model = config.get('LLM', {}).get('model', 'unknown')
llm_provider = config.get('LLM', {}).get('provider', 'unknown')
tts_method = config.get('TTS', {}).get('method', 'unknown')
stt_model = config.get('STT', {}).get('model', 'unknown')
mcp_enabled = config.get('MCP', {}).get('enabled', False)

print(f'LLM: {llm_model} ({llm_provider})')
print(f'TTS: {tts_method}')
print(f'STT: {stt_model}')
print(f'MCP: {\"enabled\" if mcp_enabled else \"disabled\"}')
print(f'Echonoti: http://localhost:9003')
" 2>/dev/null)

    if [ $? -eq 0 ] && [ -n "$CONFIG_SUMMARY" ]; then
        echo -e "${CONFIG_SUMMARY}" | while IFS= read -r line; do
            echo -e "  ${CYAN}${ARROW}${NC} ${DIM}${line}${NC}"
        done
    else
        print_info "Configuration loaded from config.yml"
        print_info "Echonoti: http://localhost:9003"
    fi
fi
set -e

# ============================================================================
# 8. Launch Services
# ============================================================================
print_section "${ROCKET}" "Launching Services"

# Start Echonoti PWA
echo -e "\n  ${CYAN}${BOLD}${GLOBE} Starting Echonoti PWA...${NC}"
print_step "Running on http://localhost:9003"

cd echonoti
npm run dev > /tmp/echonoti.log 2>&1 &
ECHONOTI_PID=$!
cd ..

# Wait a moment for Echonoti to start
sleep 2

# Check if Echonoti is running
if kill -0 $ECHONOTI_PID 2>/dev/null; then
    print_success "Echonoti PWA started (PID: $ECHONOTI_PID)"
else
    print_error "Failed to start Echonoti PWA"
    echo -e "  ${DIM}Check /tmp/echonoti.log for details${NC}"
    exit 1
fi

# Start EchoSee
echo -e "\n  ${GREEN}${BOLD}${ROCKET} Starting EchoSee voice assistant...${NC}"
echo -e "  ${DIM}Press Ctrl+C to stop all services${NC}\n"
echo -e "${DIM}════════════════════════════════════════════════════════════════${NC}\n"

# Launch the main application
python -m app_manager.manager &
ECHOSEE_PID=$!

# Wait for the main process
wait $ECHOSEE_PID

# Note: cleanup() will be called automatically via trap on exit
