#!/bin/bash

# EchoSee Launch Script
# ======================
# Complete launch script with environment setup, dependency checks,
# MCP server health checks, and beautiful output.

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

# Parse command-line arguments
SKIP_DEPS=false
SKIP_HEALTH=false
SKIP_ENV=false
SKIP_ALL=false

show_help() {
    echo -e "${CYAN}${BOLD}EchoSee Launch Script${NC}"
    echo -e "${DIM}Usage: ./launch.sh [OPTIONS]${NC}\n"
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
    exit 1
}

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
# 2. Virtual Environment
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
# 5. MCP Server Health Check
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
# 6. Configuration Summary
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
" 2>/dev/null)

    if [ $? -eq 0 ] && [ -n "$CONFIG_SUMMARY" ]; then
        echo -e "${CONFIG_SUMMARY}" | while IFS= read -r line; do
            echo -e "  ${CYAN}${ARROW}${NC} ${DIM}${line}${NC}"
        done
    else
        print_info "Configuration loaded from config.yml"
    fi
fi
set -e

# ============================================================================
# 7. Launch Application
# ============================================================================
print_section "${ROCKET}" "Launching EchoSee"

echo -e "\n  ${GREEN}${BOLD}${ROCKET} Starting voice assistant...${NC}"
echo -e "  ${DIM}Press Ctrl+C to stop${NC}\n"
echo -e "${DIM}════════════════════════════════════════════════════════════════${NC}\n"

# Launch the main application
python -m app_manager.manager || handle_error "Application runtime"

# Cleanup on exit
echo -e "\n${DIM}════════════════════════════════════════════════════════════════${NC}"
echo -e "\n${CYAN}${INFO} EchoSee stopped gracefully${NC}\n"
