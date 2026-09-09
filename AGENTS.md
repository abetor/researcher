# AGENTS.md - tool-researcher

## Purpose and boundaries

This repository contains a stateless, resumable deep-research engine. It turns a
research question or an authorized transcript dump into sources, claims, evidence,
verification decisions, and rendered wiki pages. The package is `researcher`, and
the console entry point is `research`.

The repository contains code, tests, and public documentation only. Topic data,
transcripts, generated pages, logs, receipts, model sessions, and credentials belong
outside the checkout. Runtime state lives entirely in the caller-selected topic
directory. Shared topic defaults are resolved from configuration rooted at the
TOOLS_DATA environment variable; event-hook configuration is rooted at
RESEARCHER_HOME. Both roots are caller-configurable.

A real research run requires an installed and authenticated Claude Code or Codex CLI.
The only library-level tool integration is the public llmwiki package. Other tools
must be invoked through CLI contracts or the optional event hook.

## Reading order

1. Read `README.md` for the public purpose, quick start, data boundary, and limits.
2. Read `docs/DESIGN.md` for phases, ports, concurrency, configuration, and exits.
3. Read `tests/` as the executable specification.
4. Read `tests/conftest.py` before changing test isolation.
5. Consult `smoke/README.md` only for an authorized live check.

## Verification

Run the complete hermetic suite from the repository root before reporting success:

```bash
python3 -m pytest -q -p no:cacheprovider
```

Report the observed result. If a property needs a live harness or network probe and
was not exercised, say so explicitly.

## Working rules

- Tests must not contact the network, invoke a real agent harness, read the user's
  home directory, or depend on the caller's working directory. Use temporary paths,
  local fakes, and deterministic fixtures.
- Live probes belong under `smoke/`; they are not part of the hermetic test gate and
  may use only accounts, repositories, and data the operator is authorized to use.
- Never commit credentials, credential values, private hostnames, signed URLs,
  corpora, topic data, logs, generated output, or developer-machine absolute paths.
- The topic directory is the durable state machine. A vendor session identifier is
  only a resume optimization and never replaces files on disk.
- Knowledge writes go through the llmwiki contract. Direct writes to its managed
  staging or final zones are defects.
- Stop classification is owned by `researcher/adapters/base.py`; do not duplicate
  quota, transient, and fatal classification at call sites.
- Configuration parsing fails closed. Unknown keys, wrong types, unreadable explicit
  files, and conflicting path selections are errors.
- Optional features live under `researcher/ext/` and register a CLI subcommand. The
  core must not import an extension.
- `researcher/shared/` is vendored support code owned by this repository. Treat an
  edit there as a deliberate fork and explain why in the commit message.
- Keep changes minimal and preserve existing boundaries unless the task explicitly
  changes them.

## Contract changes

CLI exit codes in `researcher/cli.py`, machine-readable status fields and schema
versions, the topic layout, phase and checkpoint semantics, documented environment
variables, adapter arguments, prompt formats, and event envelopes are public
contracts. Update `README.md`, `docs/DESIGN.md`, and tests in the same change. Adapter
arguments and prompt changes also require an appropriate authorized live check or an
explicit statement that the live property remains unverified.

## Style

Use English for code, comments, documentation, diagnostics, and commit messages.
Use no emoji or dash characters in place of a plain hyphen. Keep commit subjects
short and imperative; use the body to explain why. Do not add abstractions or
configuration without a current need.

## Non-English test fixtures

Code, comments, docstrings, CLI help, and diagnostics are English. The remaining Russian strings (a few dozen lines) are deliberate: multilingual test fixtures for Cyrillic slug generation, Unicode search terms, Russian transcript segmentation, filename byte budgets, and mixed-language course metadata; a bilingual regex in the knowledge-substrate gate; and the `INDEX.md` markers of the external course-corpus format that `course_import` parses. Keep them: the characters are the subject of those tests.
