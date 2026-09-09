# researcher

[![Tests](https://github.com/abetor/researcher/actions/workflows/tests.yml/badge.svg)](https://github.com/abetor/researcher/actions/workflows/tests.yml)

`researcher` runs resumable research workflows: it turns a question or an authorized course transcript into linked sources, claims, evidence, and wiki pages. Progress lives in topic files, so a later process can resume a run after a quota or process failure.

[Quick start](#quick-start) | [Offline setup smoke](#offline-setup-smoke) | [Architecture](docs/DESIGN.md) | [Tests](#tests) | [Contributing and agent guide](AGENTS.md) | [MIT license](LICENSE)

## Problem

Long research runs fail in ordinary chat sessions: context is finite, network calls are unreliable, quotas reset later, and a second agent cannot safely resume from another vendor's session identifier. A useful research engine therefore needs durable state, explicit evidence, deterministic gates, and resumable phases outside any single LLM session.

## What it does

- creates an `llmwiki` topic from a question;
- proposes and reviews a search map;
- collects sources through interchangeable source adapters;
- extracts claims and evidence through interchangeable Claude Code or Codex CLI adapters;
- verifies, deduplicates, and promotes accepted claims;
- renders final wiki pages;
- imports user-authorized transcript dumps through the same verification pipeline;
- exposes human-readable and bounded JSON status output for schedulers;
- resumes after quota, transient, or process failures from files in the topic directory.

The repository does not contain research corpora, credentials, model sessions, generated output, or private operational handoffs.

## Architecture

The package is intentionally stateless. The topic directory is the state machine:

```text
question or transcript
        |
        v
 search map -> collect -> extract -> verify -> promote -> pages
        |          |          |          |          |
        +---------- durable topic files and checkpoint ----------+
```

`researcher/orchestrator.py` owns phase transitions. `researcher/adapters/` is the LLM harness port. `researcher/sources/` contains source adapters. `researcher/course_import.py` handles transcript imports. `researcher/pages.py` renders pages. `researcher/checkpoint.py` and `researcher/observability.py` expose durable progress. `researcher/cli.py` is the command boundary. `researcher/shared/` is vendored support code owned by this repository.

The only library-level integration is `llmwiki`, which defines and validates the topic format. Other tools are invoked through command-line contracts or optional event hooks, never imported as sibling source trees. See [docs/DESIGN.md](docs/DESIGN.md) for the detailed contracts.

## Quick start

Python 3.11 or newer is required. Obtain [llm-wiki](https://github.com/abetor/llm-wiki)
as a sibling checkout named `../tool-llm-wiki`
(`git clone https://github.com/abetor/llm-wiki ../tool-llm-wiki`), then install it before this package.
The repository is `researcher`, the Python package is `researcher`, and the CLI is `research`.

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e ../tool-llm-wiki
python3 -m pip install -e .
python3 -m researcher doctor
```

Create a topic without starting an LLM harness:

```bash
research_root="$(mktemp -d)"
python3 -m researcher start "How should retry budgets be designed?" \
  --base "$research_root" --no-run --json > "$research_root/start.json"
topic_dir="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["topic_dir"])' "$research_root/start.json")"
```

Use the returned `topic_dir`; directory names include a hash suffix.
Run or resume that topic with an installed, authenticated harness:

```bash
python3 -m researcher resume "$topic_dir"
python3 -m researcher status "$topic_dir" --short
```

Use `python3 -m researcher --help` and the subcommand help for the full interface.

Generated prompts ask the model to answer in Russian by default. That is a product default of
the original deployment, not a translation leftover; edit the prompt strings in
`researcher/pages.py`, `researcher/report.py`, and `researcher/course_import.py` for English output.

## Offline setup smoke

This smoke creates a clean topic without network or credentials. It checks installation and
topic layout only: no collect, extract, verify, promote, or pages phase runs, and it does not
exercise resume after a failure. Those phases are covered by the test suite with fake harnesses.

```bash
demo_root="$(mktemp -d)"
python3 -m researcher start "Public release smoke test" --base "$demo_root" --no-run
python3 -m researcher status --base "$demo_root"
```

Expected result: `start` prints the topic path and `topic ready, checkpoint planned`; `status`
lists the topic in phase `planned` with zero sources and claims, reading only files under
`demo_root`.

## Data and credential boundary

Topic data is always selected explicitly with `--base` or a topic path. When `--base` is omitted, the CLI reads `paths.topics_root` from the shared configuration rooted at `TOOLS_DATA`; its documented fallback is `~/tools-data`. The researcher-specific event-hook configuration is rooted at `RESEARCHER_HOME`; its fallback is `~/tools-data/researcher-data`. Both environment variables accept any user-selected path, so the defaults are conveniences rather than hard-coded deployment requirements.

Credentials are read only from the process environment. Supported names include `GITHUB_TOKEN`, `EXA_API_KEY`, and `JINA_API_KEY`; endpoints and source policy use `RESEARCHER_SEARXNG_URL`, `RESEARCHER_WEB_BACKEND`, and `RESEARCHER_FETCH_FALLBACKS`. Leave credential variables unset to use keyless sources. Never commit `.env` files, signed URLs, transcript dumps, topic directories, logs, receipts, or generated pages.

## Limitations

- A real research run requires an installed and authenticated Claude Code or Codex CLI. The test suite replaces both with fakes.
- Source quality and availability are external inputs. The engine records evidence and stop reasons but cannot make an unreliable source authoritative.
- `llmwiki` is a runtime dependency; use the sibling editable checkout above. The current
  `doctor` location check expects that layout and may warn about otherwise importable
  installations elsewhere. This release does not claim a standalone package-index setup.
- Live harness and network checks under `smoke/` are deliberately outside the hermetic test gate.
- Test fixtures include Russian and mixed-language text where multilingual behavior itself is under test: Cyrillic slug generation, Unicode search terms, transcript segmentation, filename byte budgets.
- Generated prompts ask the model to answer in Russian by default; see the quick start.

## Tests

Run the complete hermetic suite without bytecode or pytest caches:

```bash
python3 -m pip install 'pytest>=8'
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
```

Run the offline environment check and clean-topic smoke separately:

```bash
python3 -m researcher doctor
demo_root="$(mktemp -d)"
python3 -m researcher start "Public release smoke test" --base "$demo_root" --no-run
python3 -m researcher status --base "$demo_root" --json
```

## Provenance

This repository began as a public source snapshot of a personal tool. Earlier local development
history is not included.

## License

MIT. Copyright (c) 2026 abetor. See [LICENSE](LICENSE).
