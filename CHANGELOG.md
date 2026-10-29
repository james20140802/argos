# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses simple `MAJOR.MINOR.PATCH` version tags (not strict SemVer
guarantees, since Argos is a single-operator local tool).

This history was reconstructed retroactively from annotated release tags
(`git tag -n99`) and their commit ranges; entries before this file existed may be
less granular than future ones.

## [0.7.1] — 2026-10-29

Re-summarizing old events no longer stalls on one-article events with English
headline titles. 2 commits since v0.7.0.

### Fixed
- `argos resummarize-events` no longer skips one-article events whose title
  pieces ("Build Datasets", "Letting Large Models Debate") were demanded
  verbatim in the Korean summary. Real product and company names (ChatGPT,
  GPT-4.1, Hugging Face) must still stay in English, including when the title is
  the name itself. The check only gets narrower; no name that was optional
  becomes required (ARG-326).

## [0.7.0] — 2026-10-29

A morning push notification with today's events on each device that turns it
on, and a page per technology with its events in time order. 8 commits since
v0.6.0.

**Upgrade note:**

- One new Alembic migration (push subscriptions and dispatch records). Run
  `uv run alembic upgrade head` from a source checkout before starting the new
  version, then restart the web app.
- Turn notifications on per device in the web app's settings ("아침 브리핑
  알림"). On iPhone this works only inside the web app added to the home
  screen (iOS 16.4+), over the Tailscale HTTPS address.
- No config or schedule change: the existing `argos brief` job sends the push
  first, then the Slack briefing as before.

### Added

- **Morning briefing push** — at the briefing time and weekdays, every device
  that turned notifications on gets one notification: the number of new events
  and the title of the top one (e.g. "오늘의 관측 7 — … 외 6건"). Tapping it
  opens the event feed. Nothing is sent when there are no new events. An event
  is "new" until a push carrying it has reached at least one device, so a
  morning when the Mac was asleep or offline is caught up on the next run, and
  events saved while a briefing was going out are not skipped. Results are
  logged to `brief.log`, and devices whose subscription expired are removed.
  `argos brief --push-only` sends only the push. The Slack briefing is
  unchanged. (ARG-321)
- **Technology page** — tapping a technology chip on an event card opens a page
  with its description, role branch path, other names and its events in time
  order. Aliases redirect to the main name. (ARG-315)

### Fixed

- `argos schedule install` waits for each job to unload before loading it again
  and keeps re-registering the remaining jobs if one fails (the always-on web
  job used to fail with exit 5 and skip the rest).

## [0.6.0] — 2026-10-25

Event pages that explain the whole event under headings fit to what kind of
event it is, technologies sorted into role branches automatically once a day,
and a login in front of the web app. 40 commits since v0.5.0.

**Upgrade note:**

- Two new Alembic migrations (event type and detail sections on events;
  technology profiles and role assignments). Run `uv run alembic upgrade head`
  from a source checkout before starting the new version.
- The web app (PWA) now asks each device to log in **once**. After upgrading,
  run `argos web token` in a terminal on the Mac and paste the token on the
  login page (or open the link from
  `argos web token --url https://<machine>.<tailnet>.ts.net` on the device).
  `config.toml` is now rewritten with `0600` permissions.
- Re-run `argos schedule install` to add the daily role assignment job
  (`com.argos.classify`, 04:30 by default; `taxonomy.assign_enabled`,
  `taxonomy.assign_time`).
- Optional one-off steps, each with a preview first (both use the local LLM and
  can take hours): `argos resummarize-events` rewrites existing events in the
  new format; `argos taxonomy assign --all` assigns every existing technology.

### Added

- **Event detail with headings** — opening an event shows a 2–3 sentence
  summary and a detailed account under headings chosen from a fixed list for its
  kind (launch / research / incident / update / adoption), written from the
  bodies of **all** linked articles. Company and product names stay in their
  original English; if a name central to the event is transliterated, the text
  is regenerated once, and on a second failure the previous name and summary
  are kept. Articles with opposing views are both covered, and very short
  articles are summarized briefly instead of padded. Measured on 7 real events:
  all succeeded on the first try, 7–39 s each. (ARG-308)
- **`argos resummarize-events [--limit] [--dry-run] [--yes]`** — applies the new
  event summary to existing events, showing how many will change before asking.
  (ARG-308)
- **Automatic role branches for technologies** — once a day, new technologies
  are assigned to a role branch without approval, using an LLM description plus
  evidence from articles (names listed side by side, "X is a …" phrases, GitHub
  topics and Hugging Face tags of linked repositories). Generic words, places
  and company names are not assigned; technologies with too little evidence stay
  unclassified. `argos taxonomy assign [--all] [--name] [--dry-run]` previews and
  applies, `argos taxonomy why NAME` shows the evidence, and
  `argos taxonomy eval` scores a fixed 33-name answer set (pair F1 0.95 on a copy
  of the real database). (ARG-313)

### Changed

- **Per-article long digests are no longer generated** — the event detail
  replaces them. An article page now shows the detail of the event it belongs
  to, with a link to the event. `argos backfill-digests` only prints a pointer to
  `resummarize-events`. (ARG-308)

### Fixed

- Vendor-prefix aliases no longer merge different companies' products or short
  generic words into one technology.
- Technologies written in lowercase (pgvector, uv, llama.cpp) are no longer
  dropped as generic words; capitalized plain words that the articles use in
  lowercase ("Agent", "Code") still are. (ARG-313)
- A temporary LLM outage during role assignment no longer stores "insufficient
  evidence" or overwrites an existing assignment; the name is retried on the
  next run. Re-running `--all` keeps external tag evidence. (ARG-313)

### Security

- **Web login + CSRF** — the web app no longer trusts "can reach the port":
  without a login cookie, pages show a login form (401), HTMX actions send the
  browser to the login page, and Keep/Pass/Untrack/settings changes are refused.
  POST requests coming from another site are rejected with 403. The token is a
  local file (`~/.config/argos/web_token`, 0600); `argos web token [--rotate]
  [--url]` shows or replaces it. The service worker never caches a login page.
  (ARG-195)
- **Secrets stay out of logs** — the DB password and Slack tokens are masked in
  `repr`/tracebacks; `config.toml` is written `0600` and new config directories
  `0700`; `argos doctor` warns (with a `chmod` hint) when `.env`, `config.toml`,
  `web_token` or the config directory is readable by other users. (ARG-195)

## [0.5.0] — 2026-10-25

Event summaries that read the article, fuller article bodies, an ambiguous-pair
check in nightly correction, and the first pieces of the technology layer (role
branches and name aliases). 53 commits since v0.4.2.

**Upgrade note:** two new Alembic migrations (role branches, name aliases). Run
`uv run alembic upgrade head` from a source checkout before starting the new
version — `argos doctor` reports when the DB is behind. Event summaries and the
pair check use `gemma4:26b` (`ollama pull gemma4:26b`; `argos doctor` warns when
it is missing). Optional one-off steps, each with a preview first:
`argos refetch-bodies --dry-run`, `argos taxonomy draft --seed` →
`argos taxonomy apply`, `argos alias-names --dry-run`.

### Added

- **Event summaries read the article body** — event names and summaries are
  written by `gemma4:26b` from the linked articles' bodies, split into chunks
  and folded when they do not fit, and the model can be changed with
  `event_summary.model`. The three largest real events finished in 17–24 s each.
  (ARG-310)
- **Fuller article bodies** — short items from RSS are completed from the
  article page while crawling; `argos body-stats [--by-host]` shows how many
  stored articles have no real body (< 500 chars) and `argos refetch-bodies`
  re-downloads them after a preview, falling back to Internet Archive snapshots
  for openai.com (at most 1 request/second). Articles that still have no body
  are marked "본문 없음" on the detail page. On the author's corpus the share of
  short bodies fell from 69.6% to 17.4%. (ARG-309)
- **Ambiguous-pair check in nightly correction** — same-site pairs whose score
  sits just above the grouping threshold are asked of a local LLM ("same
  announcement?"); only pairs judged different are split. Answers are cached so
  the same pair is never asked twice, and the correction preview marks pairs the
  LLM split. Wrong groupings fell from 33.8% to 13.6% in the judgment set, at the
  cost of keeping 30/45 correct groups intact instead of 36/45. (ARG-307)
- **`argos taxonomy`** — draft the upper branches of the technology role tree
  (`draft --seed` or LLM-suggested), rename / merge / split / delete branches in
  the draft file, and confirm once with `apply`. Confirmed branches are not
  changed by automatic processing. (ARG-311)
- **`argos alias-names`** — merges different names of the same technology
  (spacing / case variants and vendor prefixes such as "OpenAI Codex" → "Codex")
  after a preview; names whose version numbers differ are never merged, and
  `--show NAME` lists a name's other names. Stored names are not modified.
  (ARG-312)

### Changed

- **One LLM at a time, everywhere** — event summaries, the pair check and the
  taxonomy draft unload any other resident model before loading theirs and
  unload their own model afterwards, even when generation fails.
- **`argos doctor` / `argos init`** check for the summary and pair-check models.

### Fixed

- **Garbled characters in article bodies** — extraction no longer decodes UTF-8
  pages as Latin-1 ("weâ\x80\x99re" → "we’re").

### Security

- **URL validation before fetching bodies** — body refetch validates the scheme
  and blocks private-network targets before the first request, and robots.txt
  is checked on every redirect hop of an archive fetch.

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
