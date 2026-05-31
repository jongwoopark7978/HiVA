# Close VS Code window first if possible, or run this from a normal SSH terminal

# Stop remote VS Code processes
pkill -u "$USER" -f ".vscode-server" 2>/dev/null
pkill -u "$USER" -f "code-server" 2>/dev/null
pkill -u "$USER" -f "server/bin/code" 2>/dev/null

sleep 3

# Clear Codex/OpenAI extension state
rm -rf ~/.vscode-server/data/User/globalStorage/openai.chatgpt
rm -rf ~/.vscode-server/data/User/workspaceStorage/*/openai.chatgpt 2>/dev/null
rm -rf ~/.vscode-server/data/User/workspaceStorage/*/*openai* 2>/dev/null

# Clear logs
rm -rf ~/.vscode-server/data/logs/*
