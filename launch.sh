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

start_service "echonoti" npm --prefix echonoti run dev
# Launch all MCP servers defined in JSON config
while IFS=$'\t' read -r name module port; do
  start_service "${name}_mcp" uvicorn "$module" --factory --port "$port"
done < <(jq -r 'to_entries[] | "\(.key)\t\(.value.module)\t\(.value.port)"' mcp_servers/servers.json)

cleanup() {
  echo -e "${yellow}\nShutting down services...${reset}"
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait "${pids[@]}" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

echo -e "${green}All services launched. Starting main app...${reset}\n"

python -m app_manager.manager
