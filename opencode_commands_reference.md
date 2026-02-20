# OpenCode Top Commands Reference

OpenCode is a powerful local AI agent that operates through your CLI. Here is a comprehensive reference sheet of the most important commands and flags you will use to interact with it.

### Core Commands

| Command | Description |
| :--- | :--- |
| `opencode` | Start the OpenCode Terminal User Interface (TUI) in your current directory. This is the primary interactive mode. |
| `opencode [project_path]` | Start the OpenCode TUI inside a specific project path instead of the current directory. |
| `opencode run "message"` | Run OpenCode in headless mode to execute a specific prompt/task and immediately exit when finished. |

### Session Management

| Command | Description |
| :--- | :--- |
| `opencode session` | Manage existing sessions. |
| `opencode --continue` or `-c` | Automatically resume the last active session you were working on. |
| `opencode --session <id>` | Resume a specific past session using its ID. |
| `opencode --fork` | Fork the session when continuing (use alongside `--continue` or `--session` to branch a conversation without modifying history). |
| `opencode export [sessionID]` | Export your session data as a structured JSON file. |
| `opencode import <file>` | Import session data from a JSON file or remote URL. |

### Integrations & Extensibility

| Command | Description |
| :--- | :--- |
| `opencode github` | Manage the GitHub agent (authenticate and configure access to your repositories). |
| `opencode pr <number>` | Automatically fetch, checkout a specific GitHub Pull Request branch, and start OpenCode to review or edit it. |
| `opencode mcp` | Manage MCP (Model Context Protocol) servers. This is how you add external tools (databases, Slack, APIs, etc.) to OpenCode. |
| `opencode acp` | Start the ACP (Agent Client Protocol) server for deep IDE integrations (like VS Code/JetBrains). |

### Agent & Model Configuration

| Command | Description |
| :--- | :--- |
| `opencode agent` | Manage and configure custom agents (like the `@ralph` agent). |
| `opencode models` | List all available models that OpenCode can use (e.g., GPT-4, Claude, Moonshot, Llama). |
| `opencode --model <provider/model>` | Start OpenCode forcing it to use a specific model (e.g., `--model openai/gpt-4o`). |
| `opencode --agent <name>` | Start OpenCode forcing it to use a specific custom agent. |

### System Lifecycle & UI

| Command | Description |
| :--- | :--- |
| `opencode serve` | Starts a persistent, headless OpenCode server in the background (useful for API access). |
| `opencode web` | Starts the OpenCode server and automatically opens a beautiful web-based interface in your browser. |
| `opencode attach <url>` | Attach your current terminal to a running remote OpenCode server. |
| `opencode upgrade` | Upgrade OpenCode to the latest available version. |
| `opencode uninstall` | Uninstall OpenCode and firmly remove all related configuration and cache files. |

### System Insights & Troubleshooting

| Command | Description |
| :--- | :--- |
| `opencode auth` | Manage your API credentials, keys, and authentications. |
| `opencode stats` | View detailed statistics on your token usage, session durations, and total API costs. |
| `opencode db` | Access raw database tools to manage OpenCode's internal SQLite storage. |
| `opencode debug` | Launch debugging and troubleshooting tools if the CLI or an agent is misbehaving. |

### Useful CLI Flags

| Flag | Description |
| :--- | :--- |
| `--prompt <text>` | Pre-load a prompt but don't execute it immediately (useful for templating). |
| `--port <number>` | Specify the port for the server to listen on. |
| `--print-logs` | Print internal debug logs directly to standard error (`stderr`) while running. |

---

## Configuring Custom Model Providers

OpenCode supports over 75 LLMs through the `@ai-sdk/openai-compatible` interface. To configure a custom model or provider (like DeepSeek, Z.ai/GLM, or Nvidia NIM), you simply need to modify the `baseURL` and `apiKey` inside your `~/.config/opencode/opencode.json` file.

### NVIDIA NIM (Kimi, Llama 3)
```json
{
  "models": {
    "nvidia/kimi-k2.5": {
      "provider": "openai",
      "baseURL": "https://integrate.api.nvidia.com/v1",
      "apiKey": "nvapi-..." 
    }
  }
}
```

### DeepSeek API
```json
{
  "models": {
    "deepseek/deepseek-reasoner": {
      "provider": "openai",
      "baseURL": "https://api.deepseek.com/v1",
      "apiKey": "sk-..." 
    }
  }
}
```

### Z.ai (GLM Models)
```json
{
  "models": {
    "zai/glm-4": {
      "provider": "openai",
      "baseURL": "https://api.z.ai/api/paas/v4/",
      "apiKey": "your-zai-api-key" 
    }
  }
}
```

To invoke any of these models natively in OpenCode, run:
`opencode --model <key>` (e.g., `opencode --model deepseek/deepseek-reasoner`).

### Using Models with RalphFree inside OpenCode
If you've set up the `/ralph` integration in `opencode.json`, you can force the underlying RalphFree agent to execute its loops using your custom models by passing flags directly in the OpenCode chat window:

- **Z.ai (GLM-4):** `/ralph --model glm-4 "Rewrite the backend service"`
- **DeepSeek V3:** `/ralph --deepseek "Analyze this performance bottleneck"`
- **Nvidia Kimi:** `/ralph --model moonshot "Build a React component"`
