# Design

## Purpose and boundaries

`tool-researcher` is a resumable research engine. It accepts a question or an authorized transcript dump and produces a topic compatible with `llmwiki`: registered sources, claims with evidence, verification decisions, and final wiki pages.

The engine owns orchestration, checkpoints, source collection, prompt construction, harness execution, verification, promotion, page rendering, status, and outbound event-hook invocation. It does not own a scheduler, notification transport, secrets store, browser session, transcript downloader, or corpus database. Those systems meet it through files, environment variables, and CLI processes.

## Durable state machine

The topic directory is the source of truth. Vendor session identifiers are optional resume optimizations and never replace files. The main phases are:

1. `plan` - create a bounded search map and pause for review unless automatic approval was requested.
2. `collect` - run source adapters and register normalized source records.
3. `extract` - ask a harness to turn source material into candidate claims and evidence.
4. `verify` - validate structure, citations, coverage, and contradictions.
5. `promote` - move accepted claims into the final corpus through `llmwiki` contracts.
6. `pages` - render reviewable pages from the final corpus.
7. `done` - record a terminal checkpoint without deleting intermediate evidence.

Each phase writes its durable artifacts before advancing the checkpoint. A process may therefore stop after quota exhaustion, a transient provider failure, or termination and resume without trusting in-memory state.

The transcript-import path uses the same verification and page gates. It adds a deterministic import plan, document windows, an import receipt, and course-queue status derived from files rather than a separate hidden database.

## Ports

### LLM harness port

`researcher/adapters/base.py` defines the common result and stop classes: `done`, `quota`, `transient`, and `fatal`. Claude Code and Codex adapters build their own verified CLI arguments and parse their own output formats. The shared runner owns process execution, timeouts, bounded capture, and stop classification.

Quota is distinct from burst throttling. Quota waits for a reset window; throttling and network failures are transient and eligible for bounded retry. Classification order is part of the contract because a provider can return quota text with an otherwise successful process exit.

### Source port

`researcher/sources/base.py` defines normalized source records and fetch results. Built-in adapters cover web search/fetch, Hacker News, arXiv, and GitHub. A source adapter may use credentials from the environment, but the core never persists those values. Collection failures remain attributable to a source and do not silently become evidence.

### Topic format port

`llmwiki` is the sole permitted library dependency on another tool. It validates identifiers, claims, evidence references, tombstones, deduplication, and foreign-key integrity. The researcher calls this public package contract rather than importing a sibling repository by path.

### Event port

An optional `on_event` shell command receives a documented environment envelope. Delivery is best effort and cannot mutate research state or turn a successful research phase into a failure. Notification transports remain outside this repository.

## Files and concurrency

The implementation uses atomic replacement for checkpoints and other single-writer snapshots. A topic-level `resume.lock` prevents concurrent state-machine owners. Collector work may run concurrently within the configured bound; heartbeat state records active roles without becoming authoritative progress. Course-queue reservations prevent two workers from claiming the same course while allowing unrelated topics to proceed.

Machine-readable status has a schema version and byte bounds. It is reconstructed from durable artifacts so it remains valid after a restart. Human status may be richer, but both views derive from the same files.

## Configuration and data locations

Explicit CLI paths have highest precedence. Shared topic defaults come from `paths.topics_root` in `config.toml` under `TOOLS_DATA`; the fallback root is `~/tools-data`. Researcher event configuration comes from `RESEARCHER_HOME`; the fallback is `~/tools-data/researcher-data`. Callers can therefore relocate every data directory without editing source code.

Configuration parsing is fail closed. Unknown keys, invalid types, unreadable explicit files, and conflicting path selections are errors rather than ignored hints.

## Safety properties

- No credential value is written to a topic, log template, or repository fixture.
- Signed or ephemeral URLs are not fixtures.
- Prompts are written through bounded, explicit files and do not grant authority beyond the selected working directory.
- Final promotion requires structural validation and evidence links.
- Empty staging and final corpora refuse publication.
- Machine mode emits one bounded JSON object on stdout; diagnostics go to bounded per-run logs.
- Repository tests are hermetic. Live network and harness probes are opt-in scripts under `smoke/`.

## Exit contract

The CLI uses stable process codes for automation: success, plan review, quota wait, gate refusal, transient retry, and failure. Usage and configuration errors map to failure rather than introducing argparse's default code. The exact constants and JSON stop fields in `researcher/cli.py` are the executable specification.

## Extension rule

New source types implement the source port. New harnesses implement the harness port. Optional features may live under `researcher/ext/` and register a CLI subcommand. The core must not import an extension. Shared code is introduced only when a real third consumer exists; until then each repository owns its vendored support code.
