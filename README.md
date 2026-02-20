# RalphFree v1.0: The Autonomous Hybrid Agent

RalphFree is a powerful, multi-model AI coding agent designed to autonomously plan, execute, and verify complex coding tasks. It mimics the capabilities of advanced proprietary agents but runs entirely on your machine, leveraging state-of-the-art models like DeepSeek V3/R1 and Claude 3.5 Sonnet.

## Intelligence Upgrades (v1.0)

- **Dynamic Model Selector**: Switch between DeepSeek V3 (coding), R1 (reasoning), Claude 3.5 Sonnet, and GPT-4o-Mini on the fly.
- **Auto-Fallback Chain**: If one provider fails (rate limit/error), RalphFree automatically tries the next one.
- **Smart Routing**: Complex tasks (e.g., "plan this architecture") are automatically routed to reasoning models.
- **Persistent Cache**: File reads and searches are cached in SQLite to save tokens and time.
- **Plan → Execute → Reflect**: RalphFree drafts a plan, executes it, and reflects on the results before finishing.
- **Context Compression**: Automatically summarizes long conversations to stay within context limits.

## Installation

1. **Prerequisites**: Python 3.10+
   ```bash
   pip install litellm pyyaml python-dotenv
   ```

2. **Setup**:
   - Copy `.env.example` to `.env` and add your API keys.
   - Run `chmod +x ralphfree_runner.py`.

3. **Run**:
   ```bash
   ./ralphfree_runner.py --help
   ```

## Usage

### Interactive Mode (Recommended)
```bash
./ralphfree_runner.py --chat
```
- `/model <name>`: Switch models (e.g. `deepseek-chat`, `gpt-4o-mini`).
- `/cost`: Check session cost.
- `/save`: Save conversation state.

### CLI Mode (One-Shot)
```bash
# Simple task
./ralphfree_runner.py "Fix the bug in main.py"

# Force using DeepSeek V3
./ralphfree_runner.py --deepseek "Refactor the login flow"

# Use specific model
./ralphfree_runner.py --model gpt-4o-mini "Explain this code"
```

## Configuration

Edit `ralphfree_config.yaml` to customize models, costs, and agent behavior.

## OpenCode Integration

RalphFree can be integrated directly into [OpenCode AI](https://opencode.ai/) to combine OpenCode's UI/IDE ecosystem with RalphFree's strict Cognitive Planning Loop.

Add the following to your `~/.config/opencode/opencode.json` commands block:

```json
{
  "commands": {
    "ralph": {
      "description": "Invoke the RalphFree agentic engine for deep, isolated, complex multi-file planning and reasoning tasks instead of using native OpenCode execution.",
      "template": "Running RalphFree Engine with isolated planning and execution blocks:\n\n!ralphfree $ARGUMENTS\n\nPlease interpret the success/failure of the engine based on the printed logs above."
    },
    "ralph_isolated": {
      "description": "Invoke the RalphFree agentic engine securely using Git Worktree Isolation (no side-effects if unmerged).",
      "template": "Running RalphFree in an isolated Git Worktree:\n\n!ralphfree --isolate $ARGUMENTS\n\nPlease summarize the output of the isolated git task."
    }
  }
}
```

Now you can run `/ralph "complex refactor"` within OpenCode to trigger the deep-thinking local worker!

### Running Loops with Specific Models
If you want to force RalphFree to use a specific AI provider for the execution loop (like Z.ai, DeepSeek, or Kimi), you can pass the `--model` flag directly into your OpenCode chat prompt:

* **Z.ai (GLM-4):** `/ralph --model glm-4 "Build a new auth module"`
* **DeepSeek V3:** `/ralph --model deepseek-chat "Refactor the database schema"`
* **Nvidia NIM (Kimi):** `/ralph --model moonshot "Write a python script to scrape data"`

*(Ensure these models are defined in your `ralphfree_config.yaml` file first!)*
