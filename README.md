# tool-researcher

`tool-researcher` turns a research question or an authorized course transcript into a durable, auditable corpus of claims, evidence, sources, and wiki pages.

[Quick start](#quick-start) | [Offline demo](#demo) | [Architecture](docs/DESIGN.md) | [Tests](#tests) | [Contributing and agent guide](AGENTS.md) | [MIT license](LICENSE)

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

## Demo

The hermetic demo creates a clean topic without network or credentials:

```bash
demo_root="$(mktemp -d)"
python3 -m researcher start "Public release smoke test" --base "$demo_root" --no-run
python3 -m researcher status --base "$demo_root"
```

Expected result: the topic is created, its search plan waits for review, and status reads only files under `demo_root`.

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
- Code, comments, docstrings, CLI help, and diagnostics are English. The remaining Russian strings (a few dozen lines) are deliberate: multilingual test fixtures for Cyrillic slug generation, Unicode search terms, Russian transcript segmentation, filename byte budgets, and mixed-language course metadata; a bilingual regex in the knowledge-substrate gate; and the `INDEX.md` markers of the external course-corpus format that `course_import` parses.
- Generated prompts still ask the model to answer in Russian. That is a product default of the original deployment, not a leftover of translation; change it in the prompt templates if you need English output.

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

## License

MIT. Copyright (c) 2026 abetor. See [LICENSE](LICENSE).
