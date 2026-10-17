# Agent tooling

The repository includes the installed [Impeccable](https://github.com/pbakaus/impeccable) skill for Codex (`.agents/skills/impeccable`) and GitHub Copilot (`.github/skills/impeccable`), plus their companion agent definitions and UI-edit hooks. Each skill directory includes the upstream Apache 2.0 license.

The checked-in shell and Windows launchers use an installed engine or download the version pinned in `scripts/VERSION`. Downloaded platform binaries under `scripts/bin/` are local runtime files and are ignored by Git. On a fresh machine, run the launcher in a terminal with network access before using the hooks:

```sh
.agents/skills/impeccable/scripts/impeccable engine-probe
```

The launcher also accepts `IMPECCABLE_BIN` for a preinstalled engine or `IMPECCABLE_HOME` for a writable cache location. On Windows, use `impeccable.cmd` from the same scripts directory.

`.codex/hooks.json` invokes the Codex skill hook after UI edits and at the end of a turn. `.github/hooks/impeccable.json` invokes the Copilot hook after edits. Both skip execution when their skill is absent.

Shared project detector settings live in `.impeccable/config.json`. Hook consent and caches (`config.local.json` and `hook.cache.json`) stay local and are ignored.
