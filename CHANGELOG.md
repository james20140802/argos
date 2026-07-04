# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses simple `MAJOR.MINOR.PATCH` version tags (not strict SemVer
guarantees, since Argos is a single-operator local tool).

This history was reconstructed retroactively from annotated release tags
(`git tag -n99`) and their commit ranges; entries before this file existed may be
less granular than future ones.

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
