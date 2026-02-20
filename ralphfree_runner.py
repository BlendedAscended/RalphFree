#!/usr/bin/env python3
"""RalphFree Runner v1.0 — Multi-Model Agentic AI CLI with Intelligence Upgrades.

Usage:
    ralphfree "prompt"                     # Claude Pro → DeepSeek fallback
    ralphfree --deepseek "prompt"          # Force DeepSeek agentic
    ralphfree --model gpt-4o-mini "msg"    # Use specific model
    ralphfree --simple "msg"               # No tools, fast answer
    ralphfree --continue "msg"             # Continue last conversation
    ralphfree --chat                       # Interactive REPL mode
    ralphfree --models                     # List available models
    ralphfree --benchmark                  # Run benchmark tests
    ralphfree --help                       # Show help

Intelligence Features:
    - Chain-of-Thought (CoT) reasoning with <plan> tags
    - Plan → Execute → Reflect multi-stage loop
    - Smart routing (complex tasks → deepseek-reasoner)
    - Persistent SQLite cache across sessions
    - Metrics logging to .ralphfree_metrics.json
    - batch_edit and summarize_changes tools
    - Ollama local model support
    - Automatic fallback chain
"""
import subprocess
import os
import re
import sys
import glob
import json
import yaml
import time
import sqlite3
import traceback
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    import litellm
    litellm.suppress_debug_info = True
    litellm.set_verbose = False
    import logging
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)
except ImportError:
    print("[RALPHFREE] ⚠ litellm not installed. Run: pip install litellm")
    sys.exit(1)

# ─────────────────────────────────────────────
# Constants (configurable via ralphfree_config.yaml agent section)
# ─────────────────────────────────────────────
CODE_EXTENSIONS = {
    '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.go', '.rs', '.rb',
    '.php', '.c', '.cpp', '.h', '.hpp', '.cs', '.swift', '.kt',
    '.yaml', '.yml', '.json', '.toml', '.ini', '.cfg', '.conf',
    '.md', '.txt', '.rst', '.html', '.css', '.scss', '.sql',
    '.sh', '.bash', '.zsh', '.dockerfile', '.makefile',
}
MAX_FILE_SIZE = 50_000
MAX_TOTAL_INJECTION = 200_000
WORKING_DIR = os.getcwd()
HISTORY_FILE = os.path.join(WORKING_DIR, '.ralphfree_history.json')
METRICS_FILE = os.path.join(WORKING_DIR, '.ralphfree_metrics.json')
CACHE_DB = os.path.join(WORKING_DIR, '.ralphfree_cache.db')

# Defaults (overridden by config)
MAX_AGENT_TURNS = 20
COMPRESS_AFTER_TURN = 10
MAX_CHAT_TURNS_PER_MSG = 15
DEFAULT_TEMPERATURE = 0.3

# ─────────────────────────────────────────────
# Persistent SQLite Cache
# ─────────────────────────────────────────────
class PersistentCache:
    """SQLite-backed file cache that persists across sessions."""

    def __init__(self, db_path=CACHE_DB):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS file_cache (
                path TEXT PRIMARY KEY,
                size INTEGER,
                mtime REAL,
                content_hash TEXT,
                cached_at TEXT
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS search_cache (
                query TEXT,
                search_path TEXT,
                result TEXT,
                cached_at TEXT,
                PRIMARY KEY (query, search_path)
            )
        """)
        self.conn.commit()
        self.memory_cache = {}  # In-session fast lookup

    def is_cached(self, path):
        """Check if file is cached and still fresh (mtime unchanged)."""
        rel = os.path.relpath(path, WORKING_DIR)
        if rel in self.memory_cache:
            return True
        try:
            cur = self.conn.execute("SELECT mtime FROM file_cache WHERE path = ?", (rel,))
            row = cur.fetchone()
            if row and os.path.exists(path):
                current_mtime = os.path.getmtime(path)
                if abs(current_mtime - row[0]) < 0.01:
                    self.memory_cache[rel] = True
                    return True
        except Exception:
            pass
        return False

    def mark_cached(self, path, size):
        """Mark a file as cached."""
        rel = os.path.relpath(path, WORKING_DIR)
        self.memory_cache[rel] = size
        try:
            mtime = os.path.getmtime(path) if os.path.exists(path) else 0
            self.conn.execute(
                "INSERT OR REPLACE INTO file_cache (path, size, mtime, cached_at) VALUES (?, ?, ?, ?)",
                (rel, size, mtime, datetime.now().isoformat())
            )
            self.conn.commit()
        except Exception:
            pass

    def get_cached_search(self, query, search_path):
        """Get cached search result if still fresh (< 5 min)."""
        try:
            cur = self.conn.execute(
                "SELECT result, cached_at FROM search_cache WHERE query = ? AND search_path = ?",
                (query, search_path)
            )
            row = cur.fetchone()
            if row:
                cached_at = datetime.fromisoformat(row[1])
                if (datetime.now() - cached_at).seconds < 300:
                    return row[0]
        except Exception:
            pass
        return None

    def cache_search(self, query, search_path, result):
        """Cache a search result."""
        try:
            self.conn.execute(
                "INSERT OR REPLACE INTO search_cache (query, search_path, result, cached_at) VALUES (?, ?, ?, ?)",
                (query, search_path, result, datetime.now().isoformat())
            )
            self.conn.commit()
        except Exception:
            pass

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass


# ─────────────────────────────────────────────
# Metrics Logger
# ─────────────────────────────────────────────
class MetricsLogger:
    """Log agent performance metrics to .ralphfree_metrics.json."""

    def __init__(self, path=METRICS_FILE):
        self.path = path
        self.sessions = []
        self._load()

    def _load(self):
        try:
            if os.path.isfile(self.path):
                with open(self.path, 'r') as f:
                    data = json.load(f)
                self.sessions = data.get('sessions', [])
        except Exception:
            self.sessions = []

    def log_session(self, model, prompt_preview, turns, tokens, cost, duration_s, status):
        entry = {
            'timestamp': datetime.now().isoformat(),
            'model': model,
            'prompt': prompt_preview[:100],
            'turns': turns,
            'tokens': tokens,
            'cost': round(cost, 6),
            'duration_s': round(duration_s, 2),
            'status': status,
            'tokens_per_turn': round(tokens / max(turns, 1), 1)
        }
        self.sessions.append(entry)
        self._save()

    def _save(self):
        try:
            with open(self.path, 'w') as f:
                json.dump({'sessions': self.sessions[-100:]}, f, indent=2)
        except Exception:
            pass

    def summary(self):
        if not self.sessions:
            return "  No metrics recorded yet."
        total = len(self.sessions)
        avg_turns = sum(s['turns'] for s in self.sessions) / total
        avg_tokens = sum(s['tokens'] for s in self.sessions) / total
        avg_cost = sum(s['cost'] for s in self.sessions) / total
        avg_dur = sum(s['duration_s'] for s in self.sessions) / total
        return (f"  Sessions: {total} | Avg turns: {avg_turns:.1f} | "
                f"Avg tokens: {avg_tokens:.0f} | Avg cost: ${avg_cost:.4f} | "
                f"Avg time: {avg_dur:.1f}s")


# ─────────────────────────────────────────────
# Project Index Generator
# ─────────────────────────────────────────────
def generate_project_index(root_dir):
    """Auto-generate a compact project index with file tree + first-line summaries."""
    SKIP_DIRS = {'.git', 'node_modules', '__pycache__', 'venv', '.env', '.venv',
                 'dist', 'build', '.next', '.cache', 'egg-info', '.DS_Store'}
    lines = [f"# Project Index: {os.path.basename(root_dir)}"]
    lines.append(f"Root: {root_dir}")
    lines.append("")
    file_count = 0
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in sorted(dirs) if d not in SKIP_DIRS and not d.startswith('.')]
        depth = os.path.relpath(root, root_dir).count(os.sep)
        if depth >= 4:
            dirs.clear()
            continue
        rel_root = os.path.relpath(root, root_dir)
        if rel_root == '.':
            rel_root = ''
        for f in sorted(files):
            if f.startswith('.') or f.endswith(('.pyc', '.pyo', '.swp', '.swo')):
                continue
            ext = os.path.splitext(f)[1].lower()
            if ext not in CODE_EXTENSIONS:
                continue
            fpath = os.path.join(root, f)
            rel_path = os.path.join(rel_root, f) if rel_root else f
            try:
                size = os.path.getsize(fpath)
                summary = ""
                with open(fpath, 'r', errors='ignore') as fh:
                    for line in fh:
                        line = line.strip()
                        if line and not line.startswith(('#!', '#', '//', '/*', '"""', "'''", '---', 'import', 'from')):
                            summary = line[:80]
                            break
                        elif line.startswith(('"""', "'''")) and len(line) > 3:
                            summary = line.strip('"\' ')[:80]
                            break
                        elif line.startswith('#') and not line.startswith('#!'):
                            summary = line.lstrip('# ')[:80]
                            break
                size_str = f"{size:,}B"
                lines.append(f"  {rel_path} ({size_str}) — {summary}" if summary else f"  {rel_path} ({size_str})")
                file_count += 1
            except Exception:
                pass
    lines.append(f"\n({file_count} files indexed)")
    return '\n'.join(lines)


def load_ralphfree_md(root_dir):
    """Load RALPHFREE.md project context file if it exists."""
    for name in ['RALPHFREE.md', 'ralphfree.md', '.ralphfree.md']:
        path = os.path.join(root_dir, name)
        if os.path.isfile(path):
            try:
                with open(path, 'r') as f:
                    content = f.read()
                if content.strip():
                    return content
            except Exception:
                pass
    return None


# ─────────────────────────────────────────────
# Conversation History
# ─────────────────────────────────────────────
def save_history(messages, provider="deepseek"):
    """Save conversation messages to .ralphfree_history.json."""
    try:
        history = {
            'timestamp': datetime.now().isoformat(),
            'provider': provider,
            'working_dir': WORKING_DIR,
            'messages': []
        }
        for msg in messages:
            role = msg.get('role', '')
            if role == 'system':
                continue
            entry = {'role': role}
            if msg.get('content'):
                content = msg['content']
                if role == 'tool' and len(content) > 2000:
                    content = content[:2000] + '... [truncated]'
                entry['content'] = content
            if msg.get('tool_calls'):
                entry['tool_calls'] = msg['tool_calls']
            if msg.get('tool_call_id'):
                entry['tool_call_id'] = msg['tool_call_id']
            history['messages'].append(entry)
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2, default=str)
        print(f"  [💾 Conversation saved — use 'ralphfree --continue' to follow up]")
    except Exception as e:
        print(f"  [⚠ Could not save history: {e}]")


def load_history():
    """Load previous conversation from .ralphfree_history.json."""
    try:
        if not os.path.isfile(HISTORY_FILE):
            print("  [⚠ No previous conversation found. Starting fresh.]")
            return None
        with open(HISTORY_FILE, 'r') as f:
            history = json.load(f)
        if history.get('working_dir') != WORKING_DIR:
            print("  [⚠ Previous conversation was in a different directory. Starting fresh.]")
            return None
        ts = history.get('timestamp', 'unknown')
        msg_count = len(history.get('messages', []))
        provider = history.get('provider', 'unknown')
        print(f"  [📂 Loaded previous conversation ({msg_count} messages, {provider}, {ts})]")
        return history
    except Exception as e:
        print(f"  [⚠ Could not load history: {e}]")
        return None


# ─────────────────────────────────────────────
# Model Selector
# ─────────────────────────────────────────────
class ModelSelector:
    """Dynamic multi-model selector with fallback chain and smart routing."""

    def __init__(self, config):
        self.models = config.get('models', {})
        self.fallback_chain = config.get('fallback_chain', ['deepseek-chat'])
        
        # Determine best default based on available keys
        self.default = config.get('default_model', 'deepseek-chat')
        
        # Check if default model's key is present
        def has_key(model_name):
            if model_name not in self.models: return False
            cfg = self.models[model_name]
            env_var = cfg.get('api_key_env', '')
            return bool(os.getenv(env_var))
            
        if not has_key(self.default):
            # Try to find a fallback that HAS a key
            for model in self.fallback_chain:
                if has_key(model):
                    # print(f"  [ℹ Auto-switching default to {model} (key found)]") # Optional logging
                    self.default = model
                    break
        
        self.current = self.default
        self._fallback_index = 0
        agent_cfg = config.get('agent', {})
        self.smart_routing = agent_cfg.get('smart_routing', False)

    def select(self, name=None):
        name = name or self.current
        if name not in self.models:
            matches = [m for m in self.models if name.lower() in m.lower()]
            if matches:
                name = matches[0]
            else:
                print(f"  [⚠ Model '{name}' not found. Using {self.default}]")
                name = self.default
        self.current = name
        return self.models[name]

    def get_litellm_model(self, name=None):
        cfg = self.select(name)
        return cfg.get('litellm_model', name or self.current)

    def get_api_key(self, name=None):
        cfg = self.select(name)
        env_var = cfg.get('api_key_env', '')
        return os.getenv(env_var, '')

    def get_api_base(self, name=None):
        cfg = self.select(name)
        return cfg.get('api_base', None)

    def get_cost(self, name=None):
        cfg = self.select(name)
        return cfg.get('cost', {'input_per_1k': 0, 'output_per_1k': 0})

    def get_temperature(self, name=None):
        cfg = self.select(name)
        return cfg.get('temperature', DEFAULT_TEMPERATURE)

    def should_use_reasoner(self, prompt, turn):
        """Smart routing: use reasoner model for complex tasks."""
        if not self.smart_routing:
            return False
        if 'deepseek-reasoner' not in self.models:
            return False
        complex_keywords = ['plan', 'architect', 'design', 'refactor', 'debug complex',
                           'optimize', 'explain why', 'root cause', 'strategy']
        prompt_lower = prompt.lower()
        if any(kw in prompt_lower for kw in complex_keywords):
            return True
        if turn > 5:
            return True
        return False

    def next_fallback(self):
        while True:
            self._fallback_index += 1
            if self._fallback_index >= len(self.fallback_chain):
                return None
            
            next_model = self.fallback_chain[self._fallback_index]
            cfg = self.models.get(next_model, {})
            env_var = cfg.get('api_key_env', '')
            
            # If model is local (no key env) or key exists, try it
            if not env_var or os.getenv(env_var):
                self.current = next_model
                return next_model
                
            print(f"  [⏩ Skipping fallback {next_model} (no API key)]")

    def reset_fallback(self):
        self._fallback_index = 0
        self.current = self.fallback_chain[0] if self.fallback_chain else self.default

    def list_models(self):
        print("\n  Available Models:")
        print("  " + "─" * 65)
        for name, cfg in self.models.items():
            env_var = cfg.get('api_key_env', '')
            has_key = bool(os.getenv(env_var, ''))
            is_local = cfg.get('litellm_model', '').startswith('ollama/')
            if is_local:
                status = "✓ (local)"
            else:
                status = "✓" if has_key else "✗ (no key)"
            marker = " ← active" if name == self.current else ""
            cost = cfg.get('cost', {})
            in_c = cost.get('input_per_1k', 0)
            out_c = cost.get('output_per_1k', 0)
            desc = cfg.get('description', '')
            print(f"  {'▸' if name == self.current else ' '} {name:25s} {status:15s} ${in_c:.5f}/{out_c:.5f}  {desc}{marker}")
        print("  " + "─" * 65)
        print(f"  Default: {self.default}")
        print(f"  Fallback chain: {' → '.join(self.fallback_chain)}\n")


# ─────────────────────────────────────────────
# Tool Definitions (OpenAI-compatible format)
# ─────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read entire file contents. For large files, prefer view_file with line ranges.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path (relative to working dir or absolute)"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_files",
            "description": "Read multiple files at once. More efficient than calling read_file multiple times.",
            "parameters": {
                "type": "object",
                "properties": {
                    "paths": {"type": "array", "items": {"type": "string"}, "description": "List of file paths"}
                },
                "required": ["paths"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "view_file",
            "description": "View specific line range of a file with line numbers. Efficient for large files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "start_line": {"type": "integer", "description": "Start line (1-indexed). Default: 1"},
                    "end_line": {"type": "integer", "description": "End line (1-indexed). Default: 50"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write full content to a file. Creates parent dirs. Use edit_file for small changes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "content": {"type": "string", "description": "Full file content"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Find and replace text in a file. Surgical — only changes matched text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "old_text": {"type": "string", "description": "Exact text to find (must match including whitespace)"},
                    "new_text": {"type": "string", "description": "Replacement text"}
                },
                "required": ["path", "old_text", "new_text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "batch_edit",
            "description": "Apply multiple edits across one or more files in a single call. Efficient for multi-file changes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "edits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "old_text": {"type": "string"},
                                "new_text": {"type": "string"}
                            },
                            "required": ["path", "old_text", "new_text"]
                        },
                        "description": "Array of {path, old_text, new_text} edit operations"
                    }
                },
                "required": ["edits"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_changes",
            "description": "Show a diff summary of recent changes to a file (git diff or content comparison).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to summarize changes for"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List directory contents. The project index already has the tree — use sparingly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path. '.' for current."},
                    "recursive": {"type": "boolean", "description": "List recursively (max 3 levels). Default: false."}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for text pattern across files using grep. Use FIRST to find relevant code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Text or regex pattern"},
                    "path": {"type": "string", "description": "Directory or file to search. Default: current dir."},
                    "file_pattern": {"type": "string", "description": "Glob filter (e.g., '*.py'). Optional."}
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command. Use for tests, git ops, installs, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"}
                },
                "required": ["command"]
            }
        }
    }
]


# ─────────────────────────────────────────────
# Tool Executors
# ─────────────────────────────────────────────
def resolve_path(path):
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        path = os.path.join(WORKING_DIR, path)
    return os.path.normpath(path)

def suggest_similar_files(path):
    """Suggest similar files when a path is not found (anti-hallucination)."""
    dirname = os.path.dirname(path)
    basename = os.path.basename(path)
    if not os.path.isdir(dirname):
        dirname = WORKING_DIR
    try:
        files = os.listdir(dirname)
        matches = [f for f in files if basename.lower() in f.lower() or
                   os.path.splitext(basename)[0].lower() in f.lower()]
        if matches:
            return f" Did you mean: {', '.join(matches[:5])}?"
    except Exception:
        pass
    return ""


def tool_read_file(path, cache=None):
    path = resolve_path(path)
    rel = os.path.relpath(path, WORKING_DIR)
    if cache and cache.is_cached(path):
        return f"[CACHED — already in your context] {rel}"
    try:
        size = os.path.getsize(path)
        if size > MAX_FILE_SIZE:
            return f"Error: File too large ({size} bytes). Use view_file with line ranges."
        with open(path, 'r', errors='ignore') as f:
            content = f.read()
        if cache:
            cache.mark_cached(path, size)
        return f"── {rel} ({size} bytes) ──\n{content}"
    except FileNotFoundError:
        hint = suggest_similar_files(path)
        return f"Error: File not found: {path}.{hint}"
    except Exception as e:
        return f"Error reading {path}: {e}"


def tool_read_files(paths, cache=None):
    results = []
    total_size = 0
    for p in paths:
        if total_size > MAX_TOTAL_INJECTION:
            results.append("\n--- (skipped remaining: size limit) ---")
            break
        content = tool_read_file(p, cache)
        results.append(content)
        total_size += len(content)
    return '\n\n'.join(results)


def tool_view_file(path, start_line=1, end_line=50, cache=None):
    path = resolve_path(path)
    rel = os.path.relpath(path, WORKING_DIR)
    try:
        with open(path, 'r', errors='ignore') as f:
            all_lines = f.readlines()
        total = len(all_lines)
        start = max(1, start_line) - 1
        end = min(total, end_line)
        selected = all_lines[start:end]
        numbered = [f"{i+start+1:4d}: {line.rstrip()}" for i, line in enumerate(selected)]
        if cache:
            cache.mark_cached(path, os.path.getsize(path))
        return f"── {rel} (lines {start+1}-{end} of {total}) ──\n" + '\n'.join(numbered)
    except FileNotFoundError:
        hint = suggest_similar_files(path)
        return f"Error: File not found: {path}.{hint}"
    except Exception as e:
        return f"Error viewing {path}: {e}"


def tool_write_file(path, content):
    path = resolve_path(path)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write(content)
        return f"✓ Written {len(content)} bytes to {os.path.relpath(path, WORKING_DIR)}"
    except Exception as e:
        return f"Error writing {path}: {e}"


def tool_edit_file(path, old_text, new_text):
    path = resolve_path(path)
    try:
        with open(path, 'r') as f:
            content = f.read()
        if old_text not in content:
            return f"Error: Could not find the specified text in {os.path.relpath(path, WORKING_DIR)}"
        count = content.count(old_text)
        new_content = content.replace(old_text, new_text, 1)
        with open(path, 'w') as f:
            f.write(new_content)
        return f"✓ Replaced in {os.path.relpath(path, WORKING_DIR)} ({count} match(es), replaced first)"
    except FileNotFoundError:
        hint = suggest_similar_files(path)
        return f"Error: File not found: {path}.{hint}"
    except Exception as e:
        return f"Error editing {path}: {e}"


def tool_batch_edit(edits):
    """Apply multiple edits across files in one call."""
    results = []
    for i, edit in enumerate(edits):
        r = tool_edit_file(edit.get('path', ''), edit.get('old_text', ''), edit.get('new_text', ''))
        results.append(f"  [{i+1}] {r}")
    return '\n'.join(results)


def tool_summarize_changes(path):
    """Show git diff for a file."""
    path = resolve_path(path)
    rel = os.path.relpath(path, WORKING_DIR)
    try:
        result = subprocess.run(['git', 'diff', '--no-color', rel],
                                capture_output=True, text=True, timeout=10, cwd=WORKING_DIR)
        diff = result.stdout.strip()
        if diff:
            if len(diff) > 4000:
                diff = diff[:4000] + '\n... [diff truncated]'
            return f"── Changes in {rel} ──\n{diff}"
        result2 = subprocess.run(['git', 'status', '--porcelain', rel],
                                 capture_output=True, text=True, timeout=10, cwd=WORKING_DIR)
        status = result2.stdout.strip()
        if status:
            return f"── {rel}: {status} (new/modified, not yet committed) ──"
        return f"── {rel}: No changes detected ──"
    except Exception as e:
        return f"Error getting diff for {rel}: {e}"


def tool_list_dir(path, recursive=False):
    path = resolve_path(path)
    try:
        if not os.path.isdir(path):
            hint = suggest_similar_files(path)
            return f"Error: Not a directory: {path}.{hint}"
        lines = [f"📁 {os.path.relpath(path, WORKING_DIR)}/"]
        if recursive:
            for root, dirs, files in os.walk(path):
                dirs[:] = [d for d in sorted(dirs) if not d.startswith('.') and d not in {'node_modules', '__pycache__', 'venv', '.git'}]
                depth = root.replace(path, '').count(os.sep)
                if depth >= 3:
                    dirs.clear()
                    continue
                indent = "  " * (depth + 1)
                rel = os.path.relpath(root, path) if root != path else ""
                for f in sorted(files):
                    fpath = os.path.join(root, f)
                    try:
                        size = os.path.getsize(fpath)
                        relf = os.path.join(rel, f) if rel else f
                        lines.append(f"{indent}{relf} ({size:,} bytes)")
                    except:
                        pass
        else:
            entries = sorted(os.listdir(path))
            for entry in entries:
                if entry.startswith('.'):
                    continue
                full = os.path.join(path, entry)
                if os.path.isdir(full):
                    count = len([x for x in os.listdir(full) if not x.startswith('.')])
                    lines.append(f"  📁 {entry}/ ({count} items)")
                else:
                    size = os.path.getsize(full)
                    lines.append(f"  📄 {entry} ({size:,} bytes)")
        return '\n'.join(lines[:100])
    except Exception as e:
        return f"Error listing {path}: {e}"


def tool_search_files(pattern, path='.', file_pattern=None, cache=None):
    path = resolve_path(path)
    if cache:
        cached = cache.get_cached_search(pattern, path)
        if cached:
            return f"[CACHED SEARCH]\n{cached}"
    try:
        cmd = ['grep', '-rnI', '--color=never']
        if file_pattern:
            cmd.extend(['--include', file_pattern])
        for exclude in ['.git', 'node_modules', '__pycache__', 'venv', '.env']:
            cmd.extend(['--exclude-dir', exclude])
        cmd.extend([pattern, path])
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        output = result.stdout.strip()
        if not output:
            return f"No matches found for '{pattern}'"
        lines = output.split('\n')
        if len(lines) > 50:
            output = '\n'.join(lines[:50]) + f"\n... and {len(lines)-50} more matches"
        if cache:
            cache.cache_search(pattern, path, output)
        return output
    except subprocess.TimeoutExpired:
        return "Error: Search timed out"
    except Exception as e:
        return f"Error searching: {e}"


def tool_run_command(command):
    dangerous = ['rm -rf /', 'mkfs', 'dd if=', ':(){', 'fork bomb', 'rm -rf ~', 'rm -rf .']
    for d in dangerous:
        if d in command:
            return f"Error: Blocked dangerous command"
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True,
                                timeout=120, cwd=WORKING_DIR)
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += f"\n[stderr] {result.stderr}" if result.stdout else result.stderr
        if len(output) > 10000:
            output = output[:5000] + f"\n... ({len(output)-10000} chars truncated) ...\n" + output[-5000:]
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out (120s limit)"
    except Exception as e:
        return f"Error running command: {e}"


def execute_tool(name, args, cache=None):
    """Dispatch a tool call with auto-retry on read_file too large."""
    result = _execute_tool_inner(name, args, cache)
    if name == 'read_file' and 'too large' in result:
        path = args.get('path', '')
        print(f"  [🔄 Auto-retry: read_file → view_file (first 100 lines)]")
        result = _execute_tool_inner('view_file', {'path': path, 'start_line': 1, 'end_line': 100}, cache)
    return result


def _execute_tool_inner(name, args, cache=None):
    if name == "read_file":
        return tool_read_file(args.get("path", ""), cache)
    elif name == "read_files":
        return tool_read_files(args.get("paths", []), cache)
    elif name == "view_file":
        return tool_view_file(args.get("path", ""), args.get("start_line", 1), args.get("end_line", 50), cache)
    elif name == "write_file":
        return tool_write_file(args.get("path", ""), args.get("content", ""))
    elif name == "edit_file":
        return tool_edit_file(args.get("path", ""), args.get("old_text", ""), args.get("new_text", ""))
    elif name == "batch_edit":
        return tool_batch_edit(args.get("edits", []))
    elif name == "summarize_changes":
        return tool_summarize_changes(args.get("path", ""))
    elif name == "list_dir":
        return tool_list_dir(args.get("path", "."), args.get("recursive", False))
    elif name == "search_files":
        return tool_search_files(args.get("pattern", ""), args.get("path", "."), args.get("file_pattern"), cache)
    elif name == "run_command":
        return tool_run_command(args.get("command", ""))
    else:
        return f"Error: Unknown tool '{name}'"


# ─────────────────────────────────────────────
# Context Compression (aggressive — after 10 turns)
# ─────────────────────────────────────────────
def compress_context(messages):
    """Aggressively summarize old tool results to save tokens."""
    if len(messages) < 8:
        return messages
    compressed = []
    cutoff = len(messages) - 6
    edits_done = []
    files_read = []
    commands_run = []
    searches_done = []

    for i, msg in enumerate(messages):
        if i < 2:
            compressed.append(msg)
            continue
        if i < cutoff:
            role = msg.get('role', '')
            if role == 'tool':
                content = msg.get('content', '')
                if '✓ Replaced' in content or '✓ Written' in content:
                    edits_done.append(content.split('\n')[0][:80])
                elif content.startswith('──'):
                    fname = content.split('(')[0].strip('── ').strip()
                    files_read.append(fname)
                elif 'exit code' in content or content.startswith('$'):
                    commands_run.append(content[:60])
                elif 'matches' in content.lower():
                    searches_done.append(content.split('\n')[0][:60])
                compressed.append({
                    'role': 'tool',
                    'tool_call_id': msg.get('tool_call_id', ''),
                    'content': '[compressed]'
                })
            elif role == 'assistant' and msg.get('tool_calls'):
                compressed.append(msg)
            else:
                compressed.append(msg)
        else:
            compressed.append(msg)

    summary_parts = []
    if files_read:
        summary_parts.append(f"Files read: {', '.join(set(files_read))}")
    if edits_done:
        summary_parts.append(f"Edits done: {'; '.join(edits_done)}")
    if commands_run:
        summary_parts.append(f"Commands: {'; '.join(commands_run)}")
    if searches_done:
        summary_parts.append(f"Searches: {'; '.join(searches_done)}")

    if summary_parts:
        compressed.insert(0, {
            'role': 'user',
            'content': "[CONTEXT SUMMARY — earlier results compressed]\n" + '\n'.join(summary_parts)
        })
    return compressed


# ─────────────────────────────────────────────
# System Prompt Builder (with CoT + guardrails)
# ─────────────────────────────────────────────
def build_system_prompt(model_name, max_turns=MAX_AGENT_TURNS, chat_mode=False):
    """Build system prompt with CoT reasoning, tool guardrails, and project context."""
    project_index = generate_project_index(WORKING_DIR)
    ralph_md = load_ralphfree_md(WORKING_DIR)
    ralph_md_section = ""
    if ralph_md:
        ralph_md_section = f"\n## PROJECT RULES (from RALPHFREE.md — follow these)\n{ralph_md}\n"

    chat_rules = ""
    if chat_mode:
        chat_rules = """
## CHAT MODE RULES
6. STOP after completing the task — do NOT keep reading extra files or docs.
7. Once edits are done, give a brief summary and STOP. Do not verify by reading unrelated files.
8. You have at most 15 tool-calling turns per message. Budget wisely.
9. Respond concisely unless asked for details.
10. If the user asks a question, answer directly — don't go exploring."""

    stop_phrase_instr = "TASK COMPLETE"

    return f"""You are an expert agentic coding assistant with full filesystem access.
Working directory: {WORKING_DIR}
Model: {model_name}

## CHAIN-OF-THOUGHT REASONING
Before any action, output a step-by-step plan in <plan> tags:
<plan>
1. Understand the task and identify what needs to change
2. List the specific files and tools needed
3. Execute with minimal tool calls (batch when possible)
4. Verify the result
</plan>
Then proceed with tool calls or your final response. Always plan first.

## PROJECT INDEX (already scanned — DO NOT call list_dir on root)
{project_index}
{ralph_md_section}
## HOW TO WORK

### Step 1: PLAN (think before acting)
Before making any edits, first state your plan:
- Which files need to change
- What changes to make in each file
- In what order
Do NOT start editing until you have a clear plan.

### Step 2: EXECUTE (make changes efficiently)
- Use search_files to locate relevant code (not read_file on every file)
- Use read_files to batch-read multiple files in one call
- Use view_file with line ranges for large files
- Use edit_file for surgical changes, write_file only for new files
- Use batch_edit for multiple changes across files in one call
- Use summarize_changes to verify edits via git diff

### Step 3: VERIFY (check your work)
- Read back edited files to confirm changes
- Run tests if applicable
- Use summarize_changes to see diffs

## TOOL SELECTION GUARDRAILS
- Choose the FEWEST tools possible. Batch actions (e.g., read_files for multiple).
- Avoid redundant reads — files are cached after first read (re-read returns [CACHED]).
- If searching, use specific patterns with file_pattern filters.
- Use batch_edit instead of multiple edit_file calls.
- Do NOT call list_dir on root — the project index already has the tree.

## CONTEXT MANAGEMENT
- If context is growing large, previous tool results are automatically compressed.
- Summarize what you've learned so far if switching approaches.

## EFFICIENCY RULES
1. You have a MAXIMUM of {max_turns} turns. Budget them wisely.
2. The project index already shows the file tree — do NOT call list_dir on root.
3. Files you've already read are cached — re-reading returns "[CACHED]".
4. PRIORITIZE: source code > config > scripts. Skip docs/READMEs unless asked.
5. Indefinite Running Mode: When task is fully done, END with the exact phrase: "{stop_phrase_instr}"
6. Always respond in English.
7. RESTRICTED CONTEXT: You are strictly forbidden from reading files unless they are explicitly listed in the SPEC.md blueprint. Do not explore the codebase. Only edit what is planned.{chat_rules}"""


# ─────────────────────────────────────────────
# LiteLLM API Caller (with temperature + Ollama support)
# ─────────────────────────────────────────────
def call_llm(model_selector, messages, tools=None, model_name=None, temperature=None):
    """Make a completion call via litellm. Returns (response_message, usage_dict, error)."""
    name = model_name or model_selector.current
    litellm_model = model_selector.get_litellm_model(name)
    api_key = model_selector.get_api_key(name)
    api_base = model_selector.get_api_base(name)
    temp = temperature if temperature is not None else model_selector.get_temperature(name)

    kwargs = {
        'model': litellm_model,
        'messages': messages,
        'temperature': temp,
        'timeout': 300,
    }
    # Ollama local models don't need API key
    if api_key and not litellm_model.startswith('ollama/'):
        kwargs['api_key'] = api_key
    if api_base:
        kwargs['api_base'] = api_base
    if tools:
        kwargs['tools'] = tools

    try:
        response = litellm.completion(**kwargs)
        choice = response.choices[0]
        usage = {
            'prompt_tokens': response.usage.prompt_tokens if response.usage else 0,
            'completion_tokens': response.usage.completion_tokens if response.usage else 0,
            'total_tokens': response.usage.total_tokens if response.usage else 0,
        }
        return choice.message, usage, None
    except Exception as e:
        return None, {}, str(e)


def calculate_cost(usage, cost_cfg):
    """Calculate cost from usage stats and cost config."""
    input_cost = (usage.get('prompt_tokens', 0) / 1000) * cost_cfg.get('input_per_1k', 0)
    output_cost = (usage.get('completion_tokens', 0) / 1000) * cost_cfg.get('output_per_1k', 0)
    return input_cost + output_cost


# ─────────────────────────────────────────────
# RalphFree Loop (with reflection, smart routing, metrics)
# ─────────────────────────────────────────────
class RalphFreeLoop:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.usage_count = 0
        self.current_provider = 'primary'
        self.selector = ModelSelector(self.config)
        self.total_tokens = 0
        self.total_cost = 0.0
        self.metrics = MetricsLogger()
        self.cache = PersistentCache()
        # Load agent settings from config
        agent_cfg = self.config.get('agent', {})
        global MAX_AGENT_TURNS, COMPRESS_AFTER_TURN, MAX_CHAT_TURNS_PER_MSG, DEFAULT_TEMPERATURE
        MAX_AGENT_TURNS = agent_cfg.get('max_turns', 20)
        COMPRESS_AFTER_TURN = agent_cfg.get('compress_after', 10)
        MAX_CHAT_TURNS_PER_MSG = agent_cfg.get('chat_turns_per_msg', 50)
        DEFAULT_TEMPERATURE = agent_cfg.get('temperature', 0.3)
        self.enable_reflection = agent_cfg.get('enable_reflection', True)
        self.max_session_tokens = agent_cfg.get('max_session_tokens', 1000000)
        self.max_session_cost = agent_cfg.get('max_session_cost', 1.0)
        self.stop_phrase = agent_cfg.get('stop_phrase', "TASK COMPLETE")

    def expand_file_refs(self, prompt):
        """Expand @file and @dir/ references in the prompt."""
        pattern = r'@((?:[/~])?[\w./*\-]+(?:/[\w./*\-]*)*)'
        matches = re.findall(pattern, prompt)
        if not matches:
            return prompt
        total_injected = 0
        injected_files = []
        for ref in matches:
            path = os.path.expanduser(ref)
            if not os.path.isabs(path):
                path = os.path.join(os.getcwd(), path)
            files_to_read = []
            if os.path.isfile(path):
                files_to_read = [path]
            elif os.path.isdir(path) or ref.endswith('/'):
                dirpath = path.rstrip('/')
                if os.path.isdir(dirpath):
                    for f in sorted(os.listdir(dirpath)):
                        fpath = os.path.join(dirpath, f)
                        if os.path.isfile(fpath) and os.path.splitext(f)[1].lower() in CODE_EXTENSIONS:
                            files_to_read.append(fpath)
            elif ref.endswith('/**'):
                dirpath = path.rstrip('*').rstrip('/')
                if os.path.isdir(dirpath):
                    for root, dirs, filenames in os.walk(dirpath):
                        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {'node_modules', '__pycache__', 'venv', '.git'}]
                        for f in sorted(filenames):
                            fpath = os.path.join(root, f)
                            if os.path.splitext(f)[1].lower() in CODE_EXTENSIONS:
                                files_to_read.append(fpath)
            elif '*' in ref:
                files_to_read = sorted(glob.glob(path))
            else:
                if os.path.isfile(path):
                    files_to_read = [path]
            if not files_to_read:
                print(f"[RALPHFREE] ⚠ @{ref} — no files found, skipping")
                continue
            file_contents = []
            for fpath in files_to_read:
                try:
                    size = os.path.getsize(fpath)
                    if size > MAX_FILE_SIZE or total_injected + size > MAX_TOTAL_INJECTION:
                        continue
                    with open(fpath, 'r', errors='ignore') as f:
                        content = f.read()
                    total_injected += len(content)
                    rel = os.path.relpath(fpath)
                    file_contents.append(f"\n--- {rel} ---\n```\n{content}\n```")
                    injected_files.append(rel)
                except Exception:
                    pass
            replacement = '\n'.join(file_contents)
            prompt = prompt.replace(f'@{ref}', replacement, 1)
        if injected_files:
            print(f"[RALPHFREE] 📎 Injected {len(injected_files)} file(s): {', '.join(injected_files[:5])}" +
                  (f" (+{len(injected_files)-5} more)" if len(injected_files) > 5 else ""))
        return prompt

    def execute_claude_pro(self, prompt):
        """Execute via Claude Pro CLI (non-interactive, agentic)."""
        try:
            result = subprocess.run(
                ['claude', '-p', '--dangerously-skip-permissions', '--max-turns', '100', prompt],
                capture_output=True, text=True, timeout=600
            )
            all_output = (result.stdout + result.stderr).lower()
            rate_limit_phrases = ['limit', 'quota', 'rate limit', 'rate-limit', 'hit your limit',
                                  'usage limit', 'exceeded', 'too many requests', '429', 'try again later',
                                  'reset', 'upgrade your plan']
            if any(phrase in all_output for phrase in rate_limit_phrases):
                if result.returncode != 0 or not result.stdout.strip():
                    return None, "RATE_LIMIT"
            if result.returncode != 0 and not result.stdout.strip():
                return None, "RATE_LIMIT"
            return result.stdout, "SUCCESS"
        except subprocess.TimeoutExpired:
            return None, "TIMEOUT"
        except FileNotFoundError:
            print("  [⚠ 'claude' CLI not found — falling back]")
            return None, "RATE_LIMIT"
        except Exception as e:
            return None, str(e)

    def execute_agentic(self, prompt, model_name=None, continue_mode=False, max_turns_override=None):
        """Execute agentic loop with Plan → Execute → Reflect stages."""
        start_time = time.time()
        model_name = model_name or self.selector.current
        cost_cfg = self.selector.get_cost(model_name)

        # Determine max turns (override > config > default)
        limit_turns = max_turns_override if max_turns_override is not None else MAX_AGENT_TURNS
        if limit_turns <= 0:
            limit_turns = float('inf')

        print(f"  [📋 Indexing project...]")
        ralph_md = load_ralphfree_md(WORKING_DIR)
        if ralph_md:
            print(f"  [📖 Found RALPHFREE.md]")

        # ─── Explicit Planning Phase ───
        # Create a separate planning call if smart routing is on or just generally good practice
        plan_content = None
        if self.selector.should_use_reasoner(prompt, 0):
            reasoner = 'deepseek-reasoner'
            print(f"  [🧠 Planner: using {reasoner} to draft initial plan]")
            plan_msgs = [{'role': 'user', 'content': f"Project Context:\n{generate_project_index(WORKING_DIR)}\n\nTask: {prompt}\n\nDraft a concise, step-by-step implementation plan (no code, just steps) in <plan> tags."}]
            plan_msg, _, _ = call_llm(self.selector, plan_msgs, model_name=reasoner)
            if plan_msg:
                plan_content = plan_msg.content
                print(f"  [📝 Plan drafted] {plan_content[:100]}...")
                
                # ENHANCEMENT 1: Physical Architect Blueprint
                spec_path = os.path.join(WORKING_DIR, 'SPEC.md')
                try:
                    with open(spec_path, 'w') as f:
                        f.write(plan_content)
                    print(f"  [📄 Plan saved to SPEC.md]")
                    
                    # Instruct executor to strictly consume SPEC.md
                    prompt += "\n\nCRITICAL INSTRUCTION: Read SPEC.md, implement the first unchecked step, check it off using the edit_file tool, and repeat until all steps are complete."
                except Exception as e:
                    print(f"  [⚠ Could not save SPEC.md: {e}]")

        system_prompt = build_system_prompt(model_name, limit_turns)

        if continue_mode:
            history = load_history()
            if history and history.get('messages'):
                messages = [{'role': 'system', 'content': system_prompt}]
                messages.extend(history['messages'])
                messages.append({'role': 'user', 'content': prompt})
                print(f"  [🔄 Continuing previous conversation]")
            else:
                messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': prompt}]
        else:
            messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': prompt}]

        if plan_content:
            messages.append({'role': 'assistant', 'content': f"I have drafted a plan:\n{plan_content}\nI will now proceed."})

        total_tokens = 0
        turns_used = 0

        while True:
            # Check Limits
            if turns_used >= limit_turns:
                 print(f"  [⚠ Hit turn limit: {limit_turns}]")
                 break
            if total_tokens >= self.max_session_tokens:
                 print(f"  [⚠ Hit token limit: {total_tokens}/{self.max_session_tokens}]")
                 break
            if self.total_cost >= self.max_session_cost:
                 print(f"  [⚠ Hit cost limit: ${self.total_cost:.2f}/${self.max_session_cost}]")
                 break

            if turns_used == COMPRESS_AFTER_TURN:
                print(f"  [🗜 Compressing context...]")
                messages = compress_context(messages)

            # Reflection (every 5 turns)
            if self.enable_reflection and turns_used > 0 and turns_used % 5 == 0:
                messages.append({
                    'role': 'user',
                    'content': '⚡ Reflect: Is the task progressing? Revise plan if needed. if stuck, try a different approach.'
                })

            message, usage, error = call_llm(self.selector, messages, tools=TOOLS, model_name=model_name)

            if error:
                print(f"  [⚠ {model_name} error: {error[:80]}]")
                next_model = self.selector.next_fallback()
                if next_model:
                    print(f"  [🔄 Falling back to {next_model}...]")
                    model_name = next_model
                    cost_cfg = self.selector.get_cost(model_name)
                    message, usage, error = call_llm(self.selector, messages, tools=TOOLS, model_name=model_name)
                    if error:
                        duration = time.time() - start_time
                        self.metrics.log_session(model_name, prompt, turns_used, total_tokens, self.total_cost, duration, "FALLBACK_FAIL")
                        return None, f"Fallback failed: {error}", total_tokens, self.total_cost
                else:
                    duration = time.time() - start_time
                    self.metrics.log_session(model_name, prompt, turns_used, total_tokens, self.total_cost, duration, "ALL_FAIL")
                    return None, error, total_tokens, self.total_cost

            # Check Stop Phrase in content (Assistant self-stop)
            if message.content and self.stop_phrase in message.content:
                 print(f"  [✓ Detected Stop Phrase: '{self.stop_phrase}']")
                 # We'll allow processing of this message (it might have final explanation) then break
                 final_stop = True
            else:
                 final_stop = False

            turn_tokens = usage.get('total_tokens', 0)
            total_tokens += turn_tokens
            self.total_tokens += turn_tokens
            turn_cost = calculate_cost(usage, cost_cfg)
            self.total_cost += turn_cost
            turns_used += 1

            msg_dict = {'role': 'assistant', 'content': message.content}
            if message.tool_calls:
                msg_dict['tool_calls'] = [
                    {'id': tc.id, 'type': 'function',
                     'function': {'name': tc.function.name, 'arguments': tc.function.arguments}}
                    for tc in message.tool_calls
                ]
            messages.append(msg_dict)

            if message.tool_calls:
                # reset stop flag if tools are called (action implies not done)
                final_stop = False 
                for tc in message.tool_calls:
                    func_name = tc.function.name
                    try:
                        func_args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        func_args = {}
                    cached = ""
                    if func_name in ('read_file', 'view_file') and 'path' in func_args:
                        p = resolve_path(func_args['path'])
                        if self.cache.is_cached(p):
                            cached = " [CACHED]"
                    print(f"  [🔧 {func_name}]{cached} {json.dumps(func_args, default=str)[:120]}")
                    result = execute_tool(func_name, func_args, self.cache)
                    messages.append({
                        'role': 'tool', 'tool_call_id': tc.id,
                        'content': str(result)[:8000]
                    })

                # Nudge if close to limit (only if not infinite)
                if limit_turns != float('inf'):
                    remaining = limit_turns - turns_used
                    if remaining == 3:
                        messages.append({
                            'role': 'user',
                            'content': '⚠ 3 turns left. Finish current edits and summarize.'
                        })
                
                print(f"  [Turn {turns_used}/{limit_turns if limit_turns != float('inf') else '∞'}] Tokens: {turn_tokens} | Cost: ${turn_cost:.6f}")
            
            else:
                # No tools called
                final_text = message.content or ''
                # If we detected stop phrase OR assistant didn't call tools (and we aren't forcing it to continue), we stop.
                # But sometimes it just talks. We'll rely on STOP_PHRASE or user interaction in interactive mode.
                # In agentic mode, traditionally no-tools means done.
                if final_stop or self.stop_phrase in final_text:
                    duration = time.time() - start_time
                    print(f"  [✓ Done in {turns_used} turn(s)] Tokens: {total_tokens} | Cost: ${self.total_cost:.6f} | Time: {duration:.1f}s")
                    if self.cache.memory_cache:
                        print(f"  [📁 Files cached: {len(self.cache.memory_cache)}]")
                    save_history(messages, provider=model_name)
                    self.metrics.log_session(model_name, prompt, turns_used, total_tokens, self.total_cost, duration, "SUCCESS")
                    return final_text, "SUCCESS", total_tokens, self.total_cost
                else:
                    # It just talked but didn't say STOP. In strict agentic mode this usually means done, 
                    # but for "Indefinite" it might be asking a question.
                    # For CLI run, we assume no-tools = done.
                    duration = time.time() - start_time
                    print(f"  [✓ Done (no tools)] Tokens: {total_tokens} | Cost: ${self.total_cost:.6f}")
                    save_history(messages, provider=model_name)
                    return final_text, "SUCCESS", total_tokens, self.total_cost

        # Break loop fallthrough
        save_history(messages, provider=model_name)
        duration = time.time() - start_time
        self.metrics.log_session(model_name, prompt, turns_used, total_tokens, self.total_cost, duration, "STOPPED")
        
        last_content = ""
        for msg in reversed(messages):
            if msg.get('role') == 'assistant' and msg.get('content'):
                last_content = msg['content']
                break
        return last_content or f"Stopped after {turns_used} turns.", "SUCCESS", total_tokens, self.total_cost

    def execute_simple(self, prompt, model_name=None):
        """Simple single-shot call (no tools). Works with any model."""
        model_name = model_name or self.selector.current
        cost_cfg = self.selector.get_cost(model_name)
        messages = [
            {'role': 'system', 'content': 'You are a helpful coding assistant. Always respond in English. Be concise.'},
            {'role': 'user', 'content': prompt}
        ]
        message, usage, error = call_llm(self.selector, messages, model_name=model_name)
        if error:
            next_model = self.selector.next_fallback()
            if next_model:
                print(f"  [🔄 Falling back to {next_model}...]")
                message, usage, error = call_llm(self.selector, messages, model_name=next_model)
                cost_cfg = self.selector.get_cost(next_model)
        if error:
            return None, error, 0, 0
        tokens = usage.get('total_tokens', 0)
        cost = calculate_cost(usage, cost_cfg)
        self.total_tokens += tokens
        self.total_cost += cost
        return message.content, "SUCCESS", tokens, cost

    def run(self, prompt, force_deepseek=False, model_name=None, agentic=True, continue_mode=False, max_turns_override=None):
        """Main execution with auto-fallback."""
        prompt = self.expand_file_refs(prompt)
        if model_name:
            self.selector.select(model_name)
            force_deepseek = True
        if self.current_provider == 'primary' and not force_deepseek:
            limit = self.config['providers']['primary']['daily_limit']
            if self.usage_count < limit:
                result, status = self.execute_claude_pro(prompt)
                if status == "SUCCESS":
                    self.usage_count += 1
                    print(f"[CLAUDE PRO] Usage: {self.usage_count}/{limit}")
                    save_history([{'role': 'user', 'content': prompt},
                                  {'role': 'assistant', 'content': result}], provider="claude")
                    return result
                elif status == "RATE_LIMIT":
                    print(f"\n  ⚡ Claude Pro rate limited — switching to {self.selector.current}...")
                    self.current_provider = 'fallback'
                    agentic = True
                elif status == "TIMEOUT":
                    print(f"\n  ⏱ Claude Pro timed out — switching to {self.selector.current}...")
                    self.current_provider = 'fallback'
                    agentic = True
                else:
                    print(f"\n  ⚠ Claude Pro error: {status} — trying {self.selector.current}...")
                    self.current_provider = 'fallback'
                    agentic = True
            else:
                print(f"[RALPHFREE] Daily limit reached ({limit}). Using {self.selector.current}...")
                self.current_provider = 'fallback'

        active_model = model_name or self.selector.current
        if agentic:
            temp = self.selector.get_temperature(active_model)
            limit_str = f"{max_turns_override}" if max_turns_override else f"{MAX_AGENT_TURNS} (default)"
            if max_turns_override == float('inf') or (max_turns_override is None and MAX_AGENT_TURNS == float('inf')):
                limit_str = "∞"
            print(f"[RALPHFREE 🤖 AGENTIC v1.0] Model: {active_model} | Max turns: {limit_str} | Temp: {temp}")
            result, status, tokens, cost = self.execute_agentic(prompt, active_model, continue_mode=continue_mode, max_turns_override=max_turns_override)
        else:
            print(f"[RALPHFREE 💬 SIMPLE] Model: {active_model}")
            result, status, tokens, cost = self.execute_simple(prompt, active_model)

        if status == "SUCCESS":
            print(f"[RALPHFREE] Total tokens: {tokens} | Cost: ${cost:.6f}")
            return result
        else:
            raise Exception(f"All providers failed. Last error: {status}")

    # ─────────────────────────────────────────────
    # Interactive Chat Mode
    # ─────────────────────────────────────────────
    def run_interactive(self, model_name=None, max_turns_override=None):
        """Interactive REPL mode with persistent conversation."""
        active_model = model_name or self.selector.current
        self.selector.select(active_model)
        cost_cfg = self.selector.get_cost(active_model)

        # Determine effective turns limit per message for chat
        chat_limit = MAX_CHAT_TURNS_PER_MSG
        if max_turns_override is not None:
             chat_limit = max_turns_override if max_turns_override > 0 else 999999

        print(f"  [📋 Indexing project...]")
        system_prompt = build_system_prompt(active_model, chat_limit, chat_mode=True)
        ralph_md = load_ralphfree_md(WORKING_DIR)
        if ralph_md:
            print(f"  [📖 Found RALPHFREE.md]")

        messages = [{'role': 'system', 'content': system_prompt}]
        total_tokens = 0
        total_cost = 0.0
        turn_count = 0

        print(f"\n{'='*55}")
        print(f"  RALPHFREE CHAT v1.0 — Interactive Mode")
        print(f"  Model: {active_model} | Temp: {self.selector.get_temperature(active_model)}")
        print(f"  Turns per msg: {'∞' if chat_limit > 10000 else chat_limit}")
        print(f"  Commands: /exit /clear /help /save /model /models /cost /benchmark /metrics")
        print(f"{'='*55}\n")

        while True:
            try:
                user_input = input("\033[1;36mralphfree> \033[0m").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n\n👋 Goodbye!")
                save_history(messages, provider=active_model)
                break

            if not user_input:
                continue

            if user_input.startswith('/'):
                parts = user_input.split(maxsplit=1)
                cmd = parts[0].lower()

                if cmd in ('/exit', '/quit', '/q'):
                    print("👋 Goodbye!")
                    save_history(messages, provider=active_model)
                    break
                elif cmd == '/clear':
                    messages = [{'role': 'system', 'content': system_prompt}]
                    self.cache.memory_cache = {}
                    total_tokens = 0
                    total_cost = 0.0
                    turn_count = 0
                    print("  [🧹 Conversation cleared]\n")
                    continue
                elif cmd == '/help':
                    print("  /exit              — Exit chat")
                    print("  /clear             — Clear conversation")
                    print("  /save              — Save for --continue")
                    print("  /model <name>      — Switch model")
                    print("  /models            — List available models")
                    print("  /cost              — Show session cost")
                    print("  /metrics           — Show performance metrics")
                    print("  /benchmark         — Run benchmark tests")
                    print("  /help              — Show this help\n")
                    continue
                elif cmd == '/save':
                    save_history(messages, provider=active_model)
                    continue
                elif cmd == '/models':
                    self.selector.list_models()
                    continue
                elif cmd == '/model':
                    if len(parts) > 1:
                        new_model = parts[1].strip()
                        old_model = active_model
                        self.selector.select(new_model)
                        active_model = self.selector.current
                        cost_cfg = self.selector.get_cost(active_model)
                        system_prompt = build_system_prompt(active_model, MAX_CHAT_TURNS_PER_MSG, chat_mode=True)
                        messages[0] = {'role': 'system', 'content': system_prompt}
                        print(f"  [🔄 Switched: {old_model} → {active_model}]\n")
                    else:
                        print(f"  Current model: {active_model}")
                        print(f"  Usage: /model <name>  (e.g. /model gpt-4o-mini)\n")
                    continue
                elif cmd == '/cost':
                    print(f"  Session cost: ${total_cost:.6f}")
                    print(f"  Total tokens: {total_tokens}")
                    print(f"  Model: {active_model}\n")
                    continue
                elif cmd == '/metrics':
                    print(self.metrics.summary())
                    print()
                    continue
                elif cmd == '/benchmark':
                    self.run_benchmark()
                    continue
                else:
                    print(f"  Unknown command: {cmd}. Type /help\n")
                    continue

            user_input = self.expand_file_refs(user_input)
            messages.append({'role': 'user', 'content': user_input})

            if len(messages) > 20:
                messages = [messages[0]] + compress_context(messages[1:])

            msg_turns = 0
            for turn in range(chat_limit):
                message, usage, error = call_llm(self.selector, messages, tools=TOOLS, model_name=active_model)
                if error:
                    print(f"  [❌ API error: {error[:100]}]")
                    next_model = self.selector.next_fallback()
                    if next_model:
                        print(f"  [🔄 Falling back to {next_model}...]")
                        active_model = next_model
                        cost_cfg = self.selector.get_cost(active_model)
                        message, usage, error = call_llm(self.selector, messages, tools=TOOLS, model_name=active_model)
                        if error:
                            print(f"  [❌ Fallback failed: {error[:100]}]\n")
                            messages.pop()
                            break
                    else:
                        messages.pop()
                        break

                turn_tokens = usage.get('total_tokens', 0)
                total_tokens += turn_tokens
                turn_cost = calculate_cost(usage, cost_cfg)
                total_cost += turn_cost

                msg_dict = {'role': 'assistant', 'content': message.content}
                if message.tool_calls:
                    msg_dict['tool_calls'] = [
                        {'id': tc.id, 'type': 'function',
                         'function': {'name': tc.function.name, 'arguments': tc.function.arguments}}
                        for tc in message.tool_calls
                    ]
                messages.append(msg_dict)

                if message.tool_calls:
                    for tc in message.tool_calls:
                        func_name = tc.function.name
                        try:
                            func_args = json.loads(tc.function.arguments)
                        except json.JSONDecodeError:
                            func_args = {}
                        print(f"  [🔧 {func_name}] {json.dumps(func_args, default=str)[:100]}")
                        result = execute_tool(func_name, func_args, self.cache)
                        messages.append({
                            'role': 'tool', 'tool_call_id': tc.id,
                            'content': str(result)[:8000]
                        })
                    msg_turns += 1
                    limit_display = "∞" if chat_limit > 10000 else chat_limit
                    print(f"  [turn {msg_turns}/{limit_display}] tokens: {turn_tokens}")
                    
                    if chat_limit <= 10000 and msg_turns == chat_limit - 10:
                        messages.append({
                            'role': 'user',
                            'content': f'⚠ You use {msg_turns} turns. You have {chat_limit - msg_turns} left. Finish up.'
                        })
                else:
                    content = message.content or ''
                    turn_count += 1
                    print(f"\n{content}")
                    print(f"\n  \033[2m[{active_model} | tokens: {turn_tokens} | total: {total_tokens} | cost: ${total_cost:.4f}]\033[0m\n")
                    break

    def run_benchmark(self):
        """Run a simple benchmark to test provider latency and cost."""
        prompts = [
            ("Hello, world!", False),
            ("Write a fibonacci function in python", True)
        ]
        print("\n  🚀 Running Benchmark...")
        print("  " + "─" * 40)
        
        for name in self.selector.models:
            print(f"  Testing {name}...", end='', flush=True)
            self.selector.select(name)
            times = []
            costs = []
            
            try:
                for p, is_code in prompts:
                    t0 = time.time()
                    _, _, t, c = self.execute_simple(p, model_name=name)
                    times.append(time.time() - t0)
                    costs.append(c)
                
                avg_time = sum(times) / len(times)
                total_cost = sum(costs)
                print(f" ✓  Avg: {avg_time:.2f}s | Cost: ${total_cost:.6f}")
            except Exception as e:
                print(f" ✗  Error: {str(e)[:50]}")
        print("  " + "─" * 40 + "\n")


# ─────────────────────────────────────────────
# Worktree Automation Helpers
# ─────────────────────────────────────────────
def setup_worktree(task_name):
    """Creates a git worktree for isolated execution."""
    worktree_path = os.path.join(WORKING_DIR, '.dmux', 'worktrees', task_name)
    try:
        # Check if git is initialized
        subprocess.run(['git', 'status'], cwd=WORKING_DIR, check=True, capture_output=True)
        # Create worktree
        print(f"  [🌿 Creating isolated git worktree: {task_name}]")
        subprocess.run(['git', 'worktree', 'add', '-B', task_name, worktree_path], cwd=WORKING_DIR, check=True, capture_output=True)
        return worktree_path
    except subprocess.CalledProcessError as e:
        print(f"  [⚠ Failed to setup worktree: {e.stderr.decode() if e.stderr else str(e)}]")
        return None
    except Exception as e:
        print(f"  [⚠ Git worktree setup error: {e}]")
        return None

def merge_and_cleanup_worktree(task_name, worktree_dir):
    """Commits, merges, and cleans up the worktree."""
    print(f"  [🧹 Cleaning up and merging isolated worktree: {task_name}]")
    try:
        # Commit any changes in the worktree
        subprocess.run(['git', 'add', '.'], cwd=worktree_dir, check=True, capture_output=True)
        # We allow empty commits to not fail if the agent did nothing
        subprocess.run(['git', 'commit', '-am', f"Auto-commit from isolated task {task_name}"], cwd=worktree_dir, capture_output=True)
        
        # Switch to root directory to perform the merge
        subprocess.run(['git', 'merge', task_name, '--squash'], cwd=WORKING_DIR, capture_output=True)
        # Remove the worktree
        subprocess.run(['git', 'worktree', 'remove', '-f', worktree_dir], cwd=WORKING_DIR, check=True, capture_output=True)
        subprocess.run(['git', 'branch', '-D', task_name], cwd=WORKING_DIR, capture_output=True)
        print(f"  [✅ Successfully merged and cleaned up {task_name}]")
    except Exception as e:
        print(f"  [⚠ Merge/cleanup error for {task_name}: {e}]")
        
# ─────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    # CLI Argument Parsing (manual to avoid extra deps)
    args = sys.argv[1:]
    prompt = ""
    model = None
    simple = False
    chat = False
    isolate = False
    continue_mode = False
    deepseek = False
    show_models = False
    benchmark = False
    max_turns = None

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == '--help' or arg == '-h':
            print(__doc__)
            sys.exit(0)
        elif arg == '--model':
            model = args[i+1]
            i += 2
            continue
        elif arg == '--simple':
            simple = True
            i += 1
            continue
        elif arg == '--continue':
            continue_mode = True
            i += 1
            continue
        elif arg == '--isolate':
            isolate = True
            i += 1
            continue
        elif arg == '--chat':
            chat = True
            i += 1
            continue
        elif arg == '--deepseek':
            deepseek = True
            i += 1
            continue
        elif arg == '--models':
            show_models = True
            i += 1
            continue
        elif arg == '--benchmark':
            benchmark = True
            i += 1
            continue
        elif arg == '--max-turns' or arg == '-t' or arg == '--turns':
            if i + 1 < len(args):
                val = args[i+1]
                if val == 'inf':
                    max_turns = float('inf')
                else:
                    try:
                        max_turns = int(val)
                    except ValueError:
                        print(f"Error: Invalid max-turns value: {val}")
                        sys.exit(1)
                i += 2
            else:
                print("Error: --max-turns requires a value")
                sys.exit(1)
            continue
        elif not arg.startswith('-'):
            if prompt:
                prompt += " " + arg
            else:
                prompt = arg
            i += 1
        else:
            i += 1
            
    config_path = os.getenv('RALPHFREE_CONFIG', 'ralphfree_config.yaml')
    if not os.path.exists(config_path):
        # Fallback lookups
        for p in ['ralphfree_config.yaml', '../ralphfree_config.yaml', os.path.join(os.path.dirname(__file__), 'ralphfree_config.yaml')]:
            if os.path.exists(p):
                config_path = p
                break
                
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        print("Please create ralphfree_config.yaml")
        sys.exit(1)

    agent = RalphFreeLoop(config_path)

    if show_models:
        agent.selector.list_models()
        sys.exit(0)
        
    if benchmark:
        agent.run_benchmark()
        sys.exit(0)

    if chat:
        agent.run_interactive(model_name=model, max_turns_override=max_turns)
        sys.exit(0)

    if not prompt:
        if continue_mode:
            prompt = "Please continue."
        else:
            print("Error: No prompt provided. Use 'ralphfree --help'")
            sys.exit(1)

    worktree_dir = None
    task_name = f"isolated-task-{int(time.time())}"
    global WORKING_DIR
    original_working_dir = WORKING_DIR
    
    try:
        if isolate:
            worktree_dir = setup_worktree(task_name)
            if worktree_dir:
                os.chdir(worktree_dir)
                WORKING_DIR = worktree_dir
                print(f"  [🔒 Running isolated in {worktree_dir}]")

        result = None
        if simple:
            result, _, _, _ = agent.execute_simple(prompt, model_name=model)
        elif deepseek: # Force deepseek agentic
            result = agent.run(prompt, force_deepseek=True, model_name=model, max_turns_override=max_turns)
        else:
            result = agent.run(prompt, force_deepseek=False, model_name=model, continue_mode=continue_mode, max_turns_override=max_turns)
            
        if result:
            print(result)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        # traceback.print_exc()
        sys.exit(1)
    finally:
        if isolate and worktree_dir:
            os.chdir(original_working_dir)
            WORKING_DIR = original_working_dir
            merge_and_cleanup_worktree(task_name, worktree_dir)

if __name__ == "__main__":
    main()
