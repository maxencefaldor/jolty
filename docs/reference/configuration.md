# Configuration

Settings tune every command.
Each setting comes from the first source that sets it: a `JOLTY_*` environment variable, or the `[tool.jolty]` table of the `pyproject.toml` in the working directory.

| Setting | Environment variable | Default | Description |
| --- | --- | --- | --- |
| `timeout` | `JOLTY_TIMEOUT` | `300.0` | Seconds a script may run, its analyses included, before jolty stops it. |
| `output_limit` | `JOLTY_OUTPUT_LIMIT` | `20000` | Characters kept from each of a script's standard output and standard error. |
| `listed_unknowns` | `JOLTY_LISTED_UNKNOWNS` | `40` | Unknowns a report lists for each analysis; the results file holds every one. |

A project whose transients run long might raise the time limit and list more unknowns:

```toml title="pyproject.toml"
[tool.jolty]
timeout = 1800
listed_unknowns = 100
```
