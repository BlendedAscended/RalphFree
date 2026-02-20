# Task: Upgrade RalphFree to the Sub-Agent Architecture

You are tasked with upgrading your own codebase (`ralphfree_runner.py`) to support a safe, isolated, and scalable sub-agent architecture. This is required to prepare the tool for building a large-scale social media platform.

Please implement the following three structural enhancements. Plan carefully before making edits.

### Enhancement 1: The Physical Architect Blueprint
Right now, the `execute_agentic` function generates a plan but only keeps it in the message history. Update it to save a physical blueprint.
- Locate the "Explicit Planning Phase" inside `execute_agentic` (where it checks `self.selector.should_use_reasoner`).
- Modify the logic so that when the reasoner generates a plan, it writes that plan to a file named `SPEC.md` in the `WORKING_DIR`.
- Append explicit instructions to the `prompt` variable passed to the executor model telling it to: "Read SPEC.md, implement the first unchecked step, check it off using the edit_file tool, and repeat until all steps are complete."

### Enhancement 2: Git Worktree Isolation (Workspace Safety)
We need to stop the AI from editing live files directly. 
- Add a new helper function called `setup_worktree(task_name)` that uses `subprocess.run` to execute `git worktree add .dmux/worktrees/<task_name> -b <task_name>`.
- Add a cleanup function `merge_and_cleanup_worktree(task_name)` that commits the changes, merges the branch back to the main branch, and runs `git worktree remove`.
- Add a new CLI flag `--isolate` to the `main()` function. If this flag is passed, the tool should run the entire `execute_agentic` loop inside the generated git worktree directory instead of the main `WORKING_DIR`.

### Enhancement 3: Strict File Access Restrictions
We need to prevent the executor model from exhausting the token limit by reading the entire project.
- Modify the `build_system_prompt` function.
- Add a strict new rule under the "EFFICIENCY RULES" section: "RESTRICTED CONTEXT: You are strictly forbidden from reading files unless they are explicitly listed in the SPEC.md blueprint. Do not explore the codebase. Only edit what is planned."

### Acceptance Criteria
- `ralphfree_runner.py` parses the new `--isolate` flag without crashing.
- The Git Worktree subprocess commands are formatted correctly and safely handle errors if git is not initialized.
- The planner writes a physical `SPEC.md` file.

Execute these changes safely using your `batch_edit` or `edit_file` tools. Use `summarize_changes` to verify your work before concluding.