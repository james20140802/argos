# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses simple `MAJOR.MINOR.PATCH` version tags (not strict SemVer
guarantees, since Argos is a single-operator local tool).

This history was reconstructed retroactively from annotated release tags
(`git tag -n99`) and their commit ranges; entries before this file existed may be
less granular than future ones.

## [0.4.2] — 2026-10-22

Nightly event-boundary correction: Argos can now re-check a period's events
once a day, merging events that are really one happening and splitting ones
that mixed unrelated news. 7 commits since v0.4.1.

**Upgrade note:** correction needs the optional graph libraries. With pipx:
`pipx inject argos-scout python-igraph leidenalg` (source checkout:
`uv sync --all-extras`). Without them the correction run prints a warning and
does nothing. Preview first with `argos correct-events --dry-run`, then run
`argos schedule install` to add the daily job.

### Added

- **`argos correct-events`** — recomputes event boundaries for a period
  (`--from` / `--to`, default: the last `window_days` ending today) and applies
  the merges and splits in one transaction. `--dry-run` shows the same
  candidate report and writes nothing.
- **Daily correction schedule** — `argos schedule install` now also installs a
  daily `correct-events` job (`correction.time`, default 03:00; set
  `correction.enabled = false` to opt out). Install refuses a correction time
  that collides with the crawl or briefing times. `schedule status` and
  `schedule uninstall` include the new job.

### Changed

- **Nightly regrouping no longer splits one launch covered by several sites.**
  When the same release is written up differently by different sites (official
  post, benchmark analysis, personal take), the nightly pass used to cut it
  into pieces. Pieces from different sites whose articles are, on average,
  as related as the grouping threshold are now joined back. Pieces from a
  single site are not joined this way, so different announcements from one
  blog do not merge. Daytime grouping during crawls is unchanged. Opt out with
  `event_detection.cross_source_merge = false`.

### Fixed

- Turning the correction job off now removes it from launchd instead of
  leaving the old job behind; `argos init` no longer crashes when the schedule
  step hits a time collision, and the error says how to fix it.
- `correct-events` failures with an empty error message now show the error
  type instead of a blank line.

## [0.4.1] — 2026-10-21

Fix for `argos backfill-entities`. 1 commit since v0.4.0.

### Fixed

- **`argos backfill-entities` no longer fails with "value too long"** — an
  article whose body embedded a base64 blob produced a 13,649-character "name"
  that did not fit the name column, and the whole fill rolled back. Candidates
  longer than the stored name length (500 characters) are no longer extracted
  as names, so the fill completes. During crawling, the same article would
  previously save without any names; it now keeps its real ones. Event
  grouping is otherwise unchanged.

## [0.4.0] — 2026-10-18

The event layer: articles about the same happening are grouped into one event,
and the web feed shows events instead of individual articles. Also a
recommendation-ranked feed, a portfolio tracking timeline, and trust-score
synthesis. 233 commits since v0.3.2.

**Upgrade note:** six new Alembic migrations (event tables, document names,
SimHash, feed score, feed events, trust rubric). Run `uv run alembic upgrade head`
from a source checkout before starting the new version — `argos doctor` reports when the DB is behind.
Existing articles are not grouped into events until you run
`argos backfill-entities` and then `argos backfill-events` (preview each with
`--dry-run` first).

### Added

- **Events** — every newly saved article is assigned to an existing event or
  starts a new one, and new events get an LLM-written name. Near-duplicate
  copies of the same article (syndication, reposts) are detected and counted as
  one piece of evidence instead of many. (ARG-239, ARG-240, ARG-241, ARG-242)
- **Event feed and event pages** — the web feed lists events; articles not yet
  in an event appear as single-article items. Each event has a detail page with
  its articles, and the web UI was restyled. (ARG-243)
- **`argos recluster-events`** — regroups a period from scratch and reports
  which events look like they should be merged or split. Read-only; applying
  the corrections is not wired to a command yet. Needs the optional `graph`
  extra (`pipx inject argos-scout python-igraph leidenalg`). (ARG-244, ARG-283)
- **`argos backfill-events`** — groups articles saved before the event layer
  into events (`--dry-run` previews without writing; resumable), then
  `--rename-stale` names the events. (ARG-242)
- **`argos backfill-entities`** — re-extracts company/product names for
  existing articles, previews the change, and replaces names only after you
  answer yes. Running it twice leaves the DB unchanged. (ARG-300)
- **Company/product names per article** — extracted on crawl and used when
  deciding whether two articles belong to the same event. Generic words,
  mojibake, and names that appear across the whole corpus (`ai`, `openai`, …)
  are ignored; two articles whose names clearly differ are less likely to be
  grouped. Optional `nlp` extra adds a spaCy assist. (ARG-239, ARG-259, ARG-300)
- **Recommended feed** — the feed is sorted by a per-item score that blends
  your Keep history, trust, and freshness, with a "latest" toggle, a hero card,
  and source diversity. Impressions, clicks and dwell time are logged locally
  to improve ranking. (ARG-201, ARG-213)
- **Portfolio tracking timeline** — each kept asset shows its history and
  related signals on one timeline; a successor handoff banner lets you move
  Keep to the newer technology, and Archived items can be kept again. (ARG-199)
- **Trust score synthesis** — trust combines source prior, a rubric score, and
  cross-source corroboration; `argos backfill-trust` fills existing rows.
  (ARG-200)

### Changed

- **Day and night grouping use the same rule**, so the online assignment and
  the nightly regrouping move in the same direction when settings change.
  (ARG-282)
- **Same-source articles** published far apart are less likely to be grouped
  as one event, while same-source copies of the same news still group.
  (ARG-284, ARG-299)
- Keep/Pass and feed filters respond immediately without redrawing the whole
  page.

### Fixed

- Many name-extraction fixes (accents, initials, quotes, dots, corporate
  suffixes, `Bank of America`-style names in titles). (ARG-259, ARG-300)
- A failed event assignment no longer creates an empty event, and a failed
  link no longer reports the article as saved. (ARG-241)
- Succession alerts are sent once per asset/successor pair. (ARG-200)
- Install hints point pipx users to a command that works for them.

## [0.3.2] — 2026-07-10

Operational observability. 21 commits since v0.3.1.

### Added

- **`argos status`** — one read-only command summarising the latest scheduled
  run / daily-brief / weekly-brief outcome: last result, last success time, and
  processed counts, parsed from the launchd logs (no DB or network). (ARG-194)
- **`argos doctor` health probes** — on top of the existing Docker / Ollama /
  model / Python / uv checks, doctor now reports Postgres reachability, whether
  the applied DB revision equals the latest Alembic head, and VRAM headroom.
  (ARG-194)
- **Feed trust dial** — feed cards show the trust score as a ring dial, and the
  item-detail page shows a labeled dial. (ARG-189)

### Fixed

- **`argos status` reports the latest outcome by recency** — in the append-only
  logs an older success no longer masks a newer failure, whether that failure is
  a traceback, a non-traceback failure header, or an early config-load exit; and
  the processed counts always come from the most recent success block, never an
  older one. (ARG-220, ARG-221)
- **`argos status` recognises weekly-briefing success** — a healthy weekly run
  (and the "no items today" skip) now reads as success instead of `unknown`.
- **`argos doctor` migration probe degrades gracefully** — when argos runs from
  an installed wheel/pipx layout where the migration scripts aren't on disk, the
  Alembic probe reports a soft WARN instead of a hard FAIL, so a healthy database
  no longer makes `argos doctor` exit non-zero.

## [0.3.1] — 2026-07-07

Web polish, backup tooling, and Ollama-down resilience. 67 commits since v0.3.0.

### Added

- **Backup / restore CLI** — `argos backup` and `argos restore` for the local
  Postgres volume (ARG-192).
- **Web `/settings` page** — read/edit `config.toml` from the browser (ARG-186).
- **Item-detail actions** — Keep / Pass / Untrack buttons on the detail page,
  every transition logged to `track_history`.
- **Pull-to-refresh** — held pull-to-refresh gesture, a 새로고침 button, and a
  floating refresh control, with service-worker shell-cache sync (ARG-198).
- **Feed "새 항목" polling** — endpoint + pill UI that surfaces newer items
  without a full reload.
- **Portfolio pagination** — keyset-cursor "더 보기" fragment route so long
  portfolios load incrementally (ARG-187).
- **Typed `OllamaInfraError` + `triage_error` signal** — the brain now
  distinguishes infra outages from content-level triage results (ARG-190).

### Changed

- Refresh UX rework: self-hosted SVG icon, ring spinner, held pull gesture,
  floating button.
- `/settings` gets purpose-built controls plus a readable read-only view.

### Fixed

- **Ollama-down handling** — infra-error rows are retained in `crawl_queue`,
  are not counted as saved, `argos run` exits non-zero, and single-item `add`
  surfaces the error (ARG-190).
- Themed 500 error page via an exception handler (ARG-188).
- Feed "새 항목" pill: stays visible when a refresh fails, hides when the poll
  reports zero newer items; `htmx.process()` re-binds controls on the refreshed
  list.
- `/settings`: honor `--config` path, fix checkbox unchecks, and skip no-op
  saves by comparing against the raw on-disk value.
- Web layout: reserve scrollbar gutter on both edges to keep pages centered;
  guard native time inputs.

### Security

- Backup hardening: keep `PGPASSWORD` out of `argv` and debug logs, atomic
  restore with private dump permissions, and collision-safe concurrent dumps.

## [0.3.0] — 2026-07-02

v0.3.0 — Argos Web (PWA) & Digest Pipeline. 107 commits since v0.2.4.

### Added

- **Argos Web** — new local FastAPI web interface (PWA): `argos web` subcommand
  plus an opt-in launchd daemon.
  - Observation feed: magazine grid (featured hero + clamped summaries),
    "Midnight Observatory" design, signal ticker sidebar.
  - Keep / Pass / Untrack HTMX actions: in-place swap, toggle-off, every
    transition logged to `track_history`.
  - Item detail page: technology genealogy, pgvector similarity, `track_history`
    timeline, "new signals" section.
  - Portfolio signal/genealogy grouping, desktop multi-column grid.
  - PWA install layer (manifest / service worker / icons) + Tailscale HTTPS
    setup for iPad installs.
  - Self-hosted fonts, radar-logo SVG, light/dark theme.
- **Digest pipeline** (ARG-173, ARG-180–183) — `tech_items.digest` column plus a
  brain digest node (gate/validation), wired for both single-item and batch
  flows; `argos backfill-digests`; `argos run` progress bar now shows the
  digest stage; web detail page renders the digest lead + paragraphs; brain
  prompts now enforce configured output language.
- **Image / og:image pipeline** — `image_url` column (`tech_items`,
  `crawl_queue`) plus og:image extraction across all collection paths; priority
  image fallback chain with a favicon fallback; `argos backfill-images`
  (`--upgrade-favicons`, `--refetch`, with an SSRF gate).
- `[web]` config section (`UserConfig`) with TOML round-trip.

### Security

- SSRF gate on `backfill-images --refetch`.
- og:image now rendered via `<img src>` instead of inline CSS `url()`.

### Dependencies

- Added `fastapi`, `uvicorn`, `jinja2`, `python-multipart` (`uv.lock` updated).

### Post-deploy checklist

```bash
uv run alembic upgrade head
ollama pull qwen3:14b   # digest model — not pulled by `argos init`; backfill-digests silently fills nothing without it
uv run argos backfill-digests
uv run argos backfill-images
```

## [0.2.4] — 2026-05-31

v0.2.4 — Scout features: search, portfolio, succession & signal alerts.
102 commits since v0.2.3.

### Added

- `argos search` — semantic search over `tech_items` via pgvector cosine, Rich
  table output (ARG-64).
- `argos add <URL>` and Slack `/argos add` — manual URL injection into the
  brain pipeline (ARG-63).
- `argos portfolio` — view Keep/Tracking portfolio with category filter and
  sorting (ARG-65).
- `argos stats` — stats dashboard command (ARG-66).
- Succession alerts — detect successors for Keep-ed assets and post track
  updates to Slack (ARG-56).
- Signal matching — match new items against Keep-ed assets via pgvector
  cosine, wired into `argos run`, dispatched to Slack (ARG-55/116/117/119).
- Weekly Keep briefing — `argos brief --weekly` with a weekly launchd schedule
  and aggregated 7-day signals/successions (ARG-57).
- Live progress bar for `argos run` pipeline stages (ARG-92/114).
- Configurable genealogist model and `num_ctx`, plus a quantized-model
  benchmark harness (ARG-125/126).
- `published_at` tracking — new column, parsed from HTML meta tags / GitHub
  repo `created_at`, with a lookback filter and empty-state message
  (ARG-130/132).
- `briefed_at` dedup — items are no longer re-briefed across runs (ARG-128).
- Configurable output language, honored in triage and genealogist LLM prompts
  (ARG-127).
- HTML-clean titles and `raw_content` across HN/GitHub/arXiv/RSS, plus a
  backfill script (ARG-129).

### Fixed

- Revalidate redirected URLs in `add_url` to close an SSRF primitive.
- Per-item DB constraint isolation via savepoints; batched DB flush in the
  brain pipeline (ARG-90).
- Concurrent embed similarity searches now each use their own `AsyncSession`
  (ARG-88).
- Slack block-cap clamps for long URLs; Rich markup escaping in portfolio
  output.
- Dedupe succession alerts per asset; treat post-save dedup races as
  `DUPLICATE`, not `ERROR`.

## [0.2.3] — 2026-05-16

v0.2.3 — MVP feedback edit.

### Added

- Deep Dive language localization, Slack mrkdwn formatting, channel posting.
- Init wizard advanced-settings step (`preflight_filter`, limits).
- Generic SPA fetcher framework, with Anthropic as a default source.
- Briefing embedding-based recommendation via K-means Keep clustering.

### Fixed

- SPA fetcher: filter off-host URLs, fix async mocks, log `None` fetches.

## [0.2.2] — 2026-05-14

v0.2.2 — Crawl queue with daily processing limit (ARG-93).

After the Source Expansion release (arXiv + RSS), daily crawl volume grew to
700+ items, causing `argos run` to take 5+ hours. This release introduces a
staging queue that caps brain-pipeline processing at a configurable daily
limit (default 150 items).

### Added

- `crawl_queue` DB table — buffers all freshly crawled items; unprocessed
  items persist across runs.
- `[run] daily_limit = 150` config (`0` = unlimited); items ordered by
  `published_at DESC` so the newest tech is processed first.
- All fetchers (arXiv, RSS, HN) now emit `_published_at` for correct queue
  ordering.
- `filter_duplicate_urls` now deduplicates against both `tech_items` and
  `crawl_queue`.
- CLI output shows newly-crawled + daily-processed counts with queue
  remaining.

### Fixed

- Items where `save_node` raised a transient error are now kept in the queue
  for retry instead of being silently dropped.

Expected improvement: ~58 min/run at the default 150-item limit (vs. ~4.5h at
700+ items). Run `uv run alembic upgrade head` to create the `crawl_queue`
table.

## [0.2.1] — 2026-05-14

v0.2.1 — Brain pipeline performance improvements.

### Changed

- Batch pipeline: 3 total Ollama model swaps vs. 3×N per-URL.
- Batch embedding via `/api/embed` (single HTTP round-trip).
- Trust-score gate: skip the 32B genealogist step for low-confidence items.
- Heuristic preflight filter for job ads / marketing content.
- Config-driven `num_ctx` and genealogy context window.
- Rich summary table printed after `argos run`.

### Fixed

- Retain the `prewarm_task` reference in the `finally` block.

## [0.2.0] — 2026-05-13

v0.2.0 — Source Expansion (RSS + arXiv + Category Routing).

### Added

- arXiv fetcher for `cs.AI`/`cs.LG`/`cs.CL` abstracts.
- RSS fetcher with 8 default feeds, robots gate, User-Agent rotation.
- `source_category` hint propagated through the brain pipeline; triage now
  reads it and routes category correctly.
- `argos doctor` health probe and `--version` flag.
- Release CI workflow with PyPI Trusted Publishing.

### Fixed

- Various crawler pagination and ID normalization fixes.

## [0.1.0] — 2026-05-13

Initial tagged release.

### Changed

- PyPI distribution renamed to `argos-scout` (ARG-77).

[0.3.0]: https://github.com/james20140802/argos/releases/tag/v0.3.0
[0.2.4]: https://github.com/james20140802/argos/releases/tag/v0.2.4
[0.2.3]: https://github.com/james20140802/argos/releases/tag/v0.2.3
[0.2.2]: https://github.com/james20140802/argos/releases/tag/v0.2.2
[0.2.1]: https://github.com/james20140802/argos/releases/tag/v0.2.1
[0.2.0]: https://github.com/james20140802/argos/releases/tag/v0.2.0
[0.1.0]: https://github.com/james20140802/argos/releases/tag/v0.1.0
