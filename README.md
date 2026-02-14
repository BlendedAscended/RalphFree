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
