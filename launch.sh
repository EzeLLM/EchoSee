#!/usr/bin/env bash
set -euo pipefail

cyan="\033[0;36m"
magenta="\033[0;35m"
green="\033[0;32m"
yellow="\033[0;33m"
reset="\033[0m"

cat <<'BANNER'                                                                        
                             ,--,    ,----..                                   
    ,---,.  ,----..        ,--.'|   /   /   \   .--.--.       ,---,.    ,---,. 
  ,'  .' | /   /   \    ,--,  | :  /   .     : /  /    '.   ,'  .' |  ,'  .' | 
,---.'   ||   :     :,---.'|  : ' .   /   ;.  \  :  /`. / ,---.'   |,---.'   | 
|   |   .'.   |  ;. /|   | : _' |.   ;   /  ` ;  |  |--`  |   |   .'|   |   .' 
:   :  |-,.   ; /--` :   : |.'  |;   |  ; \ ; |  :  ;_    :   :  |-,:   :  |-, 
:   |  ;/|;   | ;    |   ' '  ; :|   :  | ; | '\  \    `. :   |  ;/|:   |  ;/| 
|   :   .'|   : |    '   |  .'. |.   |  ' ' ' : `----.   \|   :   .'|   :   .' 
|   |  |-,.   | '___ |   | :  | ''   ;  \; /  | __ \  \  ||   |  |-,|   |  |-, 
'   :  ;/|'   ; : .'|'   : |  : ; \   \  ',  / /  /`--'  /'   :  ;/|'   :  ;/| 
|   |    \'   | '/  :|   | '  ,/   ;   :    / '--'.     / |   |    \|   |    \ 
|   :   .'|   :    / ;   : ;--'     \   \ .'    `--'---'  |   :   .'|   :   .' 
|   | ,'   \   \ .'  |   ,/          `---`                |   | ,'  |   | ,'   
`----'      `---`    '---'                                `----'    `----'                                                                                    
BANNER
echo -e "${magenta}    EchoSee Launchpad${reset}\n"

# Load optional development environment
if [ -f setup_dev.sh ]; then
  source setup_dev.sh
fi

mkdir -p logs

pids=()

start_service() {
  local name="$1"
  shift
  echo -e "${cyan}▶ Starting ${name}${reset}"
  "$@" > "logs/${name}.log" 2>&1 &
  pids+=($!)
}

# Resolve python to a single environment (prefer conda env 311 if present)
PY_BIN=""
for cand in "$HOME/miniconda3/envs/311/bin/python" "$(command -v python3)" "$(command -v python)"; do
  if [ -n "$cand" ] && [ -x "$cand" ]; then
    PY_BIN="$cand"
    break
  fi
done

if [ -z "$PY_BIN" ]; then
  echo -e "${yellow}Could not determine python interpreter. Exiting.${reset}"
  exit 1
fi

echo -e "${cyan}Using Python: $PY_BIN${reset}"

# Ensure critical dependencies are present in that interpreter
echo -e "${cyan}Validating critical dependencies...${reset}"
"$PY_BIN" - <<'PY'
import sys
import subprocess

def ensure(pkg, version=None, import_name=None):
    name = import_name or pkg
    try:
        __import__(name)
    except Exception:
        target = f"{pkg}=={version}" if version else pkg
        subprocess.check_call([sys.executable, "-m", "pip", "install", target])

# MCP client/server SDK
ensure("mcp", version="1.13.0")
# Scheduler dependency for scheduler MCP
ensure("apscheduler", version="3.11.0", import_name="apscheduler")
PY

start_service "echonoti" npm --prefix echonoti run dev
# Launch all MCP servers defined in JSON config using the same interpreter
while IFS=$'\t' read -r name module port; do
  start_service "${name}_mcp" "$PY_BIN" -m uvicorn "$module" --factory --port "$port"
done < <(jq -r 'to_entries[] | "\(.key)\t\(.value.module)\t\(.value.port)"' mcp_servers/servers.json)

cleanup() {
  echo -e "${yellow}\nShutting down services...${reset}"
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait "${pids[@]}" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

echo -e "${green}All services launched. Waiting for servers to start...${reset}\n"

# Wait for MCP servers to be ready
echo -e "${cyan}Checking server health...${reset}"
for i in {1..30}; do
  all_ready=true
  while IFS=$'\t' read -r name module port; do
    if ! curl -s "http://localhost:${port}/healthz" > /dev/null 2>&1; then
      all_ready=false
      break
    fi
  done < <(jq -r 'to_entries[] | "\(.key)\t\(.value.module)\t\(.value.port)"' mcp_servers/servers.json)
  
  if [ "$all_ready" = true ]; then
    echo -e "${green}✓ All MCP servers ready!${reset}"
    break
  fi
  
  if [ $i -eq 1 ]; then
    echo -e "${yellow}Waiting for MCP servers to start...${reset}"
  fi
  
  sleep 1
done

if [ "$all_ready" != true ]; then
  echo -e "${yellow}⚠️  Some servers may not be ready, but continuing...${reset}"
fi

echo -e "${green}Starting main application...${reset}\n"

"$PY_BIN" main.py
