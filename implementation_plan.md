# Upgrade RalphFree to Sub-Agent Architecture

This plan aims to structure RalphFree for a more scalable sub-agent architecture by implementing physical blueprints, git worktree isolation, and stricter file access constraints, as outlined in `Upgrad_promtp.md`.

## Proposed Changes

### RalphFree Runner (`ralphfree_runner.py`)

#### [MODIFY] [ralphfree_runner.py](file:///Users/sandeep/Desktop/Project26/Agents/RalphFree/ralphfree_runner.py)

1. **Enhancement 1: Physical Architect Blueprint**
   - In `execute_agentic`, locate the reasoner snippet logic (around L1230). 
   - After the reasoner completes its plan (`plan_content`), use `open()` to write the plan into `SPEC.md` inside `WORKING_DIR`.
   - Concatenate the exact phrase onto the executor's `prompt`: `"Read SPEC.md, implement the first unchecked step, check it off using the edit_file tool, and repeat until all steps are complete."`

2. **Enhancement 2: Git Worktree Isolation (Workspace Safety)**
   - Introduce two helper functions globally: `setup_worktree(task_name)` and `merge_and_cleanup_worktree(task_name, worktree_dir)`.
   - `setup_worktree` will run `git worktree add .dmux/worktrees/<task_name> -b <task_name>` via `subprocess.run()`, catching errors if git is uninitialized.
   - `merge_and_cleanup_worktree` will `add`, `commit`, switch back, `merge`, and `remove` the worktree branch.
   - In `main()`, add argument parsing for `--isolate`.
   - If `--isolate` is set, dynamically generate a `task_name`, call `setup_worktree`, and temporarily update `os.chdir()` and the global `WORKING_DIR` variable to execute the agent safely. We'll use a `try...finally` block in `main()` to guarantee the worktree cleanup merges back into the main branch.

3. **Enhancement 3: Strict File Access Restrictions**
   - In `build_system_prompt()`, add a 7th rule under `## EFFICIENCY RULES`:
     `7. RESTRICTED CONTEXT: You are strictly forbidden from reading files unless they are explicitly listed in the SPEC.md blueprint. Do not explore the codebase. Only edit what is planned.`

4. **Enhancement 4: OpenCode AI Integration**
   - Provide configurations for integrating RalphFree smoothly as a custom tool within OpenCode (`~/.config/opencode/opencode.json`). 
   - Add two custom tool commands: `/ralph` and `/ralph_isolated` that drop out of OpenCode's chat window into RalphFree's strict planner agent. Example config:
     ```json
     {
       "commands": {
         "ralph": {
           "description": "Invoke the RalphFree agentic engine...",
           "template": "Running RalphFree Engine:\n\n!ralphfree $ARGUMENTS\n\nPlease interpret the success/failure logs."
         }
       }
     }
     ```
   - Provide examples of `baseURL` overrides in OpenCode for models like Z.ai (GLM), DeepSeek, and Nvidia Kimi to enhance the native architecture.

## Verification Plan

### Automated Tests
Run the updated script with `--help` to confirm it runs without syntax errors and that the flag logic parses appropriately:
```bash
./ralphfree_runner.py --help
```

### Manual Verification
1. Open the project in the terminal.
2. Run standard interaction without isolation to verify backwards compatibility:
   ```bash
   ralphfree "Write a simple text file called hello.txt"
   ```
3. Run with `--isolate` flag to verify git worktree handling creates the branch, commits, and removes the worktree automatically:
   ```bash
   ralphfree --isolate "Update hello.txt to say Hello World"
   ```
   *Verify that `.dmux/worktrees` does NOT persist and changes appear in the main branch.*
4. Ask RalphFree to perform a complex task triggering the reasoning model and observe if `SPEC.md` is effectively generated and ticked off dynamically.

## Deployment / Setup for Collaborators

To easily test or initialize this upgraded version of RalphFree on another machine (e.g., a friend's CLI), run the following commands:

### Alternative 2: Global Installation (Using `pipx` or Global Python)
If you are already inside an existing codebase (e.g., `projects/project_1`) and just want to run RalphFree without dealing with virtual environments mapped to the agent's folder, you can install its dependencies globally or via `pipx`, and move the configuration files into your local project.

```bash
# 1. Clone the project to a generic tools directory
git clone https://github.com/YourUsername/RalphFree.git ~/tools/RalphFree
cd ~/tools/RalphFree

# 2. Install dependencies globally
# (Note: Using pipx is highly recommended over global pip to avoid system conflicts)
pip install -r requirements.txt --user 
# OR if you have pipx installed: pipx install . (if setup.py exists)

# 3. Link the executable globally
chmod +x ralphfree_runner.py
mkdir -p ~/bin
ln -s "$(pwd)/ralphfree_runner.py" ~/bin/ralphfree

# --- Now, switch to your actual active project ---
cd ~/projects/project_1

# 4. Copy the agent configurations into your active project root
cp ~/tools/RalphFree/.env.example ./RALPHFREE.env
cp ~/tools/RalphFree/ralphfree_config.yaml ./
cp ~/tools/RalphFree/RALPHFREE.md ./

# (Edit RALPHFREE.env to add your API keys)

# 5. Run the agent natively inside your project!
# Tell RalphFree to use the specific .env file in your project dir
export RALPHFREE_ENV_FILE="./RALPHFREE.env" 
export RALPHFREE_CONFIG="./ralphfree_config.yaml"

ralphfree "Review the architecture of project_1"
```
