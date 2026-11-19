# Contributing

Contribution guidelines cover required checks, documentation build commands, and repository rules.

The rules live in `AGENTS.md` at the repository root, so people and every agent harness read the same rules.

The docs build with [Zensical](https://zensical.org):

```sh
uv run python docs/reference.py     # write the reference pages generated from the code
uv run python docs/illustrate.py    # draw the schematics and plots of the examples into docs/images
uv run --group docs zensical serve  # preview the site at http://localhost:8000
```

The rest of this page is `AGENTS.md`.

--8<-- "AGENTS.md:3:"
