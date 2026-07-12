# Argos Web (PWA) — `src/argos/web/`

> Scope: the local FastAPI web layer (Epic 5). This file loads when working under
> `src/argos/web/` and adds only what's specific to the web subsystem. Root
> conventions (async-first, SQLAlchemy 2.0, zero-cloud, VRAM budget, PR/Linear
> 관례, …) still apply — see the repo-root `CLAUDE.md`.

## What it is

A local FastAPI web app (PWA) for browsing the feed/portfolio from other devices on
the tailnet. Stack: **FastAPI, uvicorn, Jinja2, HTMX**. Tailscale (`tailscale serve`)
provides the HTTPS transport — the app itself speaks plain HTTP on loopback.

## Layout

```
src/argos/web/
├── app.py         # FastAPI app — routes + HTMX partial responses
├── templates/     # Jinja2 templates (full pages + HTMX fragment partials)
├── static/        # CSS / JS / static assets
├── assets/        # bundled/generated assets
└── services/      # web-layer data services (feed / portfolio / detail views)
```

Tests live at `tests/web/` (route + template tests).

## Conventions

- **Bind loopback only.** Default `127.0.0.1:8765` (`[web].host` / `[web].port` in
  `~/.config/argos/config.toml`; `--host` / `--port` override per-invocation).
  **Never bind `0.0.0.0`.** Remote access is Tailscale's job — `tailscale serve`
  terminates HTTPS and forwards to loopback — not the app's. Host is config/flag
  overridable, but the loopback-only rule stands: exposure goes through Tailscale.
- **Async-first (inherited).** FastAPI routes are `async def` and use the shared
  async session. Never introduce a sync DB call in the web layer.
- **HTMX-driven.** Interactions swap server-rendered HTML fragments; keep template
  partials renderable on their own, since they're returned directly to HTMX requests.

## Run / preview

```bash
uv run argos web [--host H] [--port P]   # default 127.0.0.1:8765
```

For a visual review from a remote device (iPad/away, can't open `localhost:8765`),
the `ui-preview` skill captures the affected routes and publishes a shareable gallery
Artifact — use it before opening a PR that changes web templates/static.
