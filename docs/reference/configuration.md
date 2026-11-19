# Configuration

Settings tune every command.
Each setting comes from the first source that sets it: a `JOLTY_*` environment variable, or the `[tool.jolty]` table of the `pyproject.toml` in the working directory.

| Setting | Environment variable | Default | Description |
| --- | --- | --- | --- |
| `timeout` | `JOLTY_TIMEOUT` | `300.0` | Maximum runtime in seconds for a script and its analyses before termination. |
| `max_output` | `JOLTY_MAX_OUTPUT` | `20000` | Maximum characters retained separately from script standard output and standard error. |
| `max_unknowns` | `JOLTY_MAX_UNKNOWNS` | `40` | Maximum unknowns listed per analysis in reports; results files store all unknowns. |
| `max_diagnostics` | `JOLTY_MAX_DIAGNOSTICS` | `20` | Maximum diagnostics listed in a check report; remaining diagnostics are counted. |
| `cache` | `JOLTY_CACHE` | Returns `$XDG_CACHE_HOME/jolty`, or `~/.cache/jolty` when unset | Directory for cached downloads such as device models; defaults to the user cache directory. |
| `pdk_root` | `JOLTY_PDK_ROOT` | unset | Directory containing open_pdks installs (sky130A, gf180mcuD, ihp-sg13g2); defaults to cached pinned releases. |
| `threads` | `JOLTY_THREADS` | unset | Worker thread count for batches and parallel analyses; defaults to one per CPU without affecting numerical results. |
| `rules` | `JOLTY_RULES` | unset | Severity override per rule name or 'all' (error, warn, ignore). Explicit rule names override 'all', and unlisted rules keep default levels. |
| `analysis_log` | `JOLTY_ANALYSIS_LOG` | unset | Append-only JSONL file recording analysis kind, run count, and start timestamp across CLI, MCP, and scripts; disabled by default. |

A project whose transients run long might raise the time limit, and one that couples stages through capacitors on purpose might report floating nodes as warnings:

```toml title="pyproject.toml"
[tool.jolty]
timeout = 1800

[tool.jolty.rules]
no-dc-path = "warn"
```
