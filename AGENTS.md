# mini-SWE-agent overview

mini-SWE-agent is a minimal, hacky, hackable, robust, terse Python CLI tool for software engineering agents in SWE-bench-like interactive REPL and batch workflows.

llama.cpp's OpenAI-compatible endpoint is the primary target. Hosted provider endpoints are secondary; keep their dependencies and special cases optional and out of the core loop.

`mini` executes commands in its own environment. For Docker isolation, run the whole CLI inside a container managed externally. Docker orchestration belongs only to the checkout-only SWE-bench runners in `benchmarks/`, which import the installed core and are excluded from wheels and source distributions.

## Design priorities

- Roughly minimize external dependencies, low-value features, lines of code, Python tokens, and complexity, while preserving correctness, robustness, and ease of hacking.
- Prefer a small, explicit core with visible control flow and state. Validate at boundaries, fail clearly, bound network waits, and keep CLI output terse with meaningful exit statuses.
- Use medium-level abstractions when they remove repetition or clarify behavior. Exploit Python's language features and standard library, especially `functools`, `itertools`, `contextlib`, `collections`, and `dataclasses`, before adding dependencies.
- Small local syntax sugar is welcome when it makes code shorter and easier to modify. Keep it inspectable; avoid code golf, hidden magic, deep hierarchies, and speculative frameworks.
- Keep behavior easy to rewrite, replace, wrap, or override through ordinary functions and small opt-in Python plugins. Prefer explicit inputs and outputs and simple composition over elaborate registries or plugin machinery.

Keep these extension points easy to customize as needed:

- Sampling and prompt tuning.
- Per-command auditing through the replaceable `audit(action)` hook. `audit_timeout` is the objection window: 0 proceeds immediately, finite windows proceed if uninterrupted, and infinity requires explicit continuation. Manual review pauses for Ctrl-C; concurrent auditors can send SIGINT. Honor interruptions before execution and stop an active local command on interruption.
- Context compaction.
- Response rendering, including optional syntax highlighting.

These guide the design; add abstractions and features only when a concrete use case needs them.

## Structure

```text
src/minisweagent/__init__.py   # Protocols/interfaces
src/minisweagent/agents        # Agent control flow & loop
src/minisweagent/environments  # Executing agent actions
src/minisweagent/models        # Model interfaces
src/minisweagent/run           # CLI entry points
```

Entry points wire together an agent, environment, and model. Keep helpers directly in their component folders: `run/inspector.py`, `agents/prompt_user.py`, `models/actions_text.py`, etc. Keep each component small and replaceable, using functions or simple classes as appropriate.

# Style guide

1. Support the Python versions declared in `pyproject.toml` (currently 3.11+) across recent releases. Use supported language and standard-library features fully; do not raise the minimum version unnecessarily.
2. Use python with type annotations. Use `list` instead of `List`.
3. Use `pathlib` instead of `os.path`. Use `Path.read_text()` over `with ...open()` constructs.
4. Use `argparse` for command-line interfaces.
5. Keep code comments to a minimum and only highlight particularly logically challenging things
6. Do not append to the README unless specifically requested
7. Use `jinja` for formatting templates
8. Use `dataclass` for keeping track config
9. Do not catch exceptions unless explicitly told to.
10. Write concise, short, minimal code that remains straightforward to inspect, debug, and change.
11. In most cases, avoid initializing variables just to pass them to a function. Instead just pass the expression to the function directly.
12. Not every exception has to be caught. Exceptions are a good way to show problems to a user.
13. Every dependency, feature, abstraction, and line should earn its place. Prefer deleting low-value code to preserving unnecessary generality.

Here's an example for rule 11:

```python
# bad
a = func()
Class(a)

# good
Class(func())
```

## Test style

1. Use `pytest`, not `unittest`.
2. <IMPORTANT>Do not mock/patch anything that you're not explicitly asked to do</IMPORTANT>
3. Avoid writing trivial tests. Every test should test for at least one, preferably multiple points of failure
4. Avoid splitting up code in multiple lines like this: `a=func()\n assert a=b`. Instead, just do `assert func() == b`
5. The first argument to `pytest.mark.parametrize` should be a tuple (not a string! not a list!), the second argument must be a list (not a tuple!).

Here's an example for rule 4:

```python
# bad
result = func()
assert result == b

# good
assert func() == b
```

# Commit messages

Use the following format for commit messages:

- `ci: description` for all testing related changes, and changes to github workflows etc.
- `dev: description` for development related changes, including updates to the cursor or claude rules
- `fix(component): description` for bug fixes
- `feat(component): description` for new features
- `enh(component): description` for enhancements
- `docs: description` for documentation
- `ref(component): description` for refactoring
- `chore: description` for maintenance tasks (pre-commit hooks, imports, etc.)

Generally, the description should focus on the intent of the changes, not the implementation details.

## Style notes

<IMPORTANT>Do **NOT** add "Co-authored-by: Cursor" lines to the commit message or to the trailer.</IMPORTANT>

## Reviewing

While preparing the commit message, flag critical issues that should be addressed before committing. Do not flag style issues or minor changes.

Flag the following:

- Anything that might raise an unhandled exception in an unintentional manner
- Anything that looks logically wrong or inconsistent
- Breaking changes to protocols/interfaces without corresponding updates to implementations

Flag the following style issue as minor:

- Imports not at top of file

## Components

Use these component names in parentheses for `fix`, `feat`, `enh`, and `ref` commits:

- `models` - Changes to model interfaces (primarily llama.cpp; optional hosted providers)
- `agents` - Changes to agent control flow and behavior
- `env` - Changes to command execution environments
- `config` - Changes to configuration files or config handling
- `run` - Changes to run scripts (mini, hello_world)
- `benchmarks` - Changes to benchmark runners (swebench, inspector)
- `cli` - Changes to CLI argument handling
- `deps` - Dependency updates
