# Benchmarks

A benchmark task gives one agent a prompt and the files it names, and the agent works on its own, as long as it needs, with jolty as its compiler: its command line, its MCP server and its Python package, and whatever code it writes beside them.
A scorer then judges what the agent left, independently of the agent and of how it worked.

| Task | The agent's objective | It meets the task when |
| --- | --- | --- |
| [`cost_down`](cost_down/task.md) | make a transistor's bias cheaper, from a catalog of resistors | its design meets the original's specs and yield and costs less |

Run an agent on a task, Claude Code unless another command is given after the task:

```sh
uv run python benchmarks/run.py benchmarks/cost_down
```

The runner copies the task into a fresh directory, runs the agent there with the task's `task.md` as its prompt, keeps its transcript, and scores the directory.
Score a directory an agent already worked in:

```sh
uv run python benchmarks/score.py benchmarks/cost_down path/to/work
```
