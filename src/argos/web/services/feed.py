"""Read-side service backing the 관측 피드 screen (ARG-155).

``fetch_feed`` returns recent ``tech_items`` joined to the user's asset
status. The service deliberately ignores the column exclusively owned by
the Slack briefing pipeline so the two surfaces stay decoupled.

ARG-213 adds a second, default sort — ``"recommended"`` (feed_score DESC,
NULLS LAST) — alongside the original ``"latest"`` (time-based) path, a
page-local same-domain-not-consecutive reorder for the recommended page, and
``select_hero`` for the magazine-hero pick. A follow-up review pass added
``latest_feed_cursor`` (the ARG-203 poll baseline must stay sort-independent)
and ``pin_hero`` (the hero must actually lead the rendered page, and
diversity reordering must not displace it).

ARG-243 moves the feed's unit from the document to the **event**. Every feed
entry is one card: a live ``tech_events`` row (its evidence documents
aggregated), or — for a document no live event claims — a one-document entry
drawn the same way. The existing sort / keyset / diversity / hero machinery is
unchanged; it now runs over entries (``_entries_subquery``) instead of raw
``tech_items`` rows:

* an entry's **representative document** is its earliest-reported evidence
  (``coalesce(published_at, created_at)`` ASC, then id) — Keep/Pass, the
  card's domain (diversity bucket) and category all come from it;
* ``sort_at`` (latest sort, poll pill) is the **newest** evidence time, so an
  event that gains a document floats up and counts as new;
* ``feed_score`` (recommended sort) is the **max** over its evidence.
"""
from __future__ import annotations

import base64
import json
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional
from urllib.parse import urlsplit

from sqlalchemy import Integer, String, exists, func, literal, select, union_all
from sqlalchemy.dialects.postgresql import aggregate_order_by
from sqlalchemy.ext.asyncio import AsyncSession

from argos.models.event_document import EventDocument
from argos.models.tech_event import TechEvent
from argos.models.tech_item import CategoryType, TechItem
from argos.models.user_asset import AssetStatus, UserAsset


PAGE_SIZE: int = 20

# ARG-213: the hero is the highest-feed_score item published/created within
# this trailing window; see ``select_hero``.
HERO_WINDOW: timedelta = timedelta(hours=48)


Category = Literal["Mainstream", "Alpha"]
FeedSort = Literal["recommended", "latest"]
EntryKind = Literal["event", "item"]


@dataclass(frozen=True)
class FeedItem:
    id: uuid.UUID
    title: str
    source_url: str
    category: Optional[CategoryType]
    image_url: Optional[str]
    summary: Optional[str]
    status: Optional[AssetStatus]
    trust_score: Optional[float]
    sort_at: datetime
    # ARG-213: carried so the "recommended" sort's keyset cursor can be
    # re-derived from the last item on a page without a second query.
    feed_score: Optional[float] = None
    # ARG-243: ``id`` is the *entry* id — the event id for ``kind="event"``,
    # the document id for a lone ``kind="item"``. Every document-level field
    # above (title fallback, source_url, category, status, trust_score) comes
    # from the representative document ``rep_id``; Keep/Pass act on it.
    kind: EntryKind = "item"
    rep_id: Optional[uuid.UUID] = None
    doc_count: int = 1
    # Distinct publisher domains across the evidence, earliest report first.
    # Empty means "just the representative's domain" (a lone document).
    source_domains: tuple[str, ...] = ()

    @property
    def action_id(self) -> uuid.UUID:
        """The document Keep/Pass/Untrack act on (ARG-243 decision 1)."""
        return self.rep_id or self.id

    @property
    def href(self) -> str:
        """Where tapping the card goes — the event page, or the old item URL."""
        if self.kind == "event":
            return f"/event/{self.id}"
        return f"/item/{self.id}"

    @property
    def source_count(self) -> int:
        """How many distinct outlets reported this (``출처 N곳``)."""
        return max(len(self.source_domains), 1)


@dataclass(frozen=True)
class FeedPage:
    items: list[FeedItem]
    next_cursor: Optional[str]


# ------------------------------------------------------------------ #
# Cursor helpers
# ------------------------------------------------------------------ #

def encode_cursor(sort_at: datetime, item_id: uuid.UUID) -> str:
    """Opaque cursor for the ``latest`` (time-based) sort.

    Tagged ``"s": "lat"`` so a ``recommended``-sort cursor accidentally fed
    into this path (or vice versa) is rejected outright — a feed_score float
    silently misread as a timestamp (or vice versa) would corrupt pagination
    instead of failing loudly (ARG-213 AC).
    """
    if sort_at.tzinfo is None:
        sort_at = sort_at.replace(tzinfo=timezone.utc)
    payload = {
        "t": sort_at.astimezone(timezone.utc).isoformat(),
        "i": item_id.hex,
        "s": "lat",
    }
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_cursor(token: str) -> tuple[datetime, uuid.UUID]:
    try:
        padded = token + "=" * (-len(token) % 4)
        raw = base64.urlsafe_b64decode(padded.encode("ascii"))
        payload = json.loads(raw.decode("utf-8"))
        if payload.get("s") != "lat":
            raise ValueError("not a latest-sort cursor")
        sort_at = datetime.fromisoformat(payload["t"])
        if sort_at.tzinfo is None:
            sort_at = sort_at.replace(tzinfo=timezone.utc)
        item_id = uuid.UUID(payload["i"])
        return sort_at, item_id
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid feed cursor: {token!r}") from exc


def encode_score_cursor(
    feed_score: Optional[float], sort_at: datetime, item_id: uuid.UUID
) -> str:
    """Opaque cursor for the ``recommended`` (feed_score) sort.

    Tagged ``"s": "rec"`` — see ``encode_cursor`` for why cross-sort cursors
    must not silently decode. ``feed_score`` may be ``None`` (the boundary row
    sits in the NULLS-LAST tail); JSON ``null`` round-trips that faithfully.

    ``sort_at`` (``coalesce(published_at, created_at)``) is the recency
    tiebreaker: the recommended order is ``feed_score DESC NULLS LAST, sort_at
    DESC, id DESC`` so that the NULL tail — every row immediately after the
    feed_score migration, and any item added between scheduled rescores —
    orders by recency instead of arbitrary UUID. The cursor therefore has to
    carry ``sort_at`` for keyset pagination to stay exact.
    """
    if sort_at.tzinfo is None:
        sort_at = sort_at.replace(tzinfo=timezone.utc)
    payload = {
        "f": feed_score,
        "t": sort_at.astimezone(timezone.utc).isoformat(),
        "i": item_id.hex,
        "s": "rec",
    }
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_score_cursor(
    token: str,
) -> tuple[Optional[float], datetime, uuid.UUID]:
    try:
        padded = token + "=" * (-len(token) % 4)
        raw = base64.urlsafe_b64decode(padded.encode("ascii"))
        payload = json.loads(raw.decode("utf-8"))
        if payload.get("s") != "rec":
            raise ValueError("not a recommended-sort cursor")
        feed_score = payload["f"]
        if feed_score is not None:
            feed_score = float(feed_score)
        sort_at = datetime.fromisoformat(payload["t"])
        if sort_at.tzinfo is None:
            sort_at = sort_at.replace(tzinfo=timezone.utc)
        item_id = uuid.UUID(payload["i"])
        return feed_score, sort_at, item_id
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid feed cursor: {token!r}") from exc


# ------------------------------------------------------------------ #
# Domain diversity (ARG-213)
# ------------------------------------------------------------------ #

def _domain_of(url: Optional[str]) -> str:
    """Normalized host, '' when unparseable/missing.

    A separate copy from ``argos.web.app``'s render-time ``_domain_of``
    filter (that one is a closure local to ``build_web_app``) so this service
    has no dependency on the app module.

    The host is lowercased and a leading ``www.`` is stripped so the value is
    a stable bucket key for ``_reorder_diverse`` (ARG-213). Without this,
    ``www.example.com`` / ``example.com`` / ``Example.com`` become distinct
    keys, so cards from the same publisher can still render consecutively
    while the same-domain constraint believes it's satisfied.
    """
    if not url:
        return ""
    try:
        host = urlsplit(url).netloc.lower()
    except ValueError:
        return ""
    if host.startswith("www."):
        host = host[4:]
    return host


def _reorder_no_adjacent(items: list, *, avoid_domain: Optional[str] = None) -> list:
    """Same-domain-not-consecutive reorder, page-local only (ARG-213).

    Repeatedly takes the next item from whichever domain currently has the
    most items *left*, skipping the domain just placed unless every
    remaining item shares it (ties broken by first-seen domain order, and
    each domain's own items are taken in their original relative order). This
    is the standard "no two adjacent equal" rearrangement — it's provably
    able to avoid every same-domain adjacency whenever the page's domain
    counts allow it (feasible iff the largest domain's count is at most half
    the page, rounded up).

    A simpler "just differ from the immediately preceding item" greedy can
    still leave an *avoidable* run near the end on a domain-skewed page: e.g.
    10 items from domain A plus 10 spread across five other domains is fully
    alternate-able, but naively draining the smaller domains first (because
    they happen to sit earlier in the page) leaves nothing but A for the
    tail. Weighting by remaining count instead avoids that.

    Only when a domain still holds more than half the *remaining* items does
    an adjacent repeat become truly unavoidable — the excess then lands
    back-to-back, in original relative order, rather than forced apart or
    shuffled. Never drops an item.

    ``avoid_domain`` seeds the "domain just placed" state so the *first*
    pick also avoids it when possible — used by ``pin_hero`` so the item
    immediately following the pinned hero doesn't accidentally share the
    hero's domain (the hero itself is never part of ``items`` here; it sits
    at index 0 and this function only ever sees the remainder).
    """
    buckets: dict[str, deque] = defaultdict(deque)
    domain_order: list[str] = []  # first-seen order, for a deterministic tie-break
    for it in items:
        domain = _domain_of(it.source_url)
        if domain not in buckets:
            domain_order.append(domain)
        buckets[domain].append(it)

    out: list = []
    last_domain: Optional[str] = avoid_domain
    total = len(items)
    while len(out) < total:
        candidates = [d for d in domain_order if buckets[d] and d != last_domain]
        if not candidates:
            # Unavoidable: every remaining item shares last_domain.
            candidates = [d for d in domain_order if buckets[d]]
        chosen_domain = max(
            candidates, key=lambda d: (len(buckets[d]), -domain_order.index(d))
        )
        chosen = buckets[chosen_domain].popleft()
        out.append(chosen)
        last_domain = chosen_domain
    return out


def _reorder_diverse(items: list, *, avoid_domain: Optional[str] = None) -> list:
    """Band-aware same-domain diversity for the recommended page (ARG-213).

    The recommended sort is ``feed_score DESC NULLS LAST``, so unscored rows
    (fresh/manual items, and everything right after the feed_score migration)
    live in a tail after all scored recommendations. Diversifying the *whole*
    page at once would let ``_reorder_no_adjacent`` pull an unscored tail row up
    between two scored cards as a same-domain spacer — e.g. ``a(0.8), a(0.7),
    b(NULL)`` → ``a, b(NULL), a`` promotes the NULL row above a real
    recommendation. So diversify the scored band and the null band
    *independently* and keep the null band strictly after the scored band; the
    scored band's final domain seeds the null band's ``avoid_domain`` so the
    boundary between them still isn't a same-domain pair. All-scored or all-null
    pages (the common cases) fall through to a single-band reorder unchanged.

    ``feed_score`` is read via ``getattr`` so the pure-function tests can pass
    lightweight item stubs without the attribute (they land in the null band).
    """
    scored = [it for it in items if getattr(it, "feed_score", None) is not None]
    nulls = [it for it in items if getattr(it, "feed_score", None) is None]
    if scored and nulls:
        scored_out = _reorder_no_adjacent(scored, avoid_domain=avoid_domain)
        null_out = _reorder_no_adjacent(
            nulls, avoid_domain=_domain_of(scored_out[-1].source_url)
        )
        return scored_out + null_out
    return _reorder_no_adjacent(items, avoid_domain=avoid_domain)


def pin_hero(
    items: list, hero_id: Optional[uuid.UUID], *, diversify: bool
) -> Optional[list]:
    """Pin the ``hero_id`` item to index 0; diversify only the remainder.

    Fix (review of ARG-213): the hero was selected by id (``select_hero``)
    but never actually moved to the front of the rendered page, so the
    full-width ``.card--featured`` hero markup and the positional tier-2 CSS
    (``argos.css`` ``nth-child(2)``/``nth-child(3)``) could land on the wrong
    card whenever the hero wasn't already first — and ``_reorder_diverse``
    (which runs over the *whole* recommended page) was free to shuffle the
    hero away from the front entirely.

    Returns ``None`` when ``hero_id`` isn't present in ``items`` at all —
    the item scored highest but simply isn't on this particular page/sort
    (e.g. a highly-scored-but-old item under ``sort="latest"``, or a
    highly-scored item that fell past this page's cursor window). Callers
    must treat ``None`` as "no pin happened" and fall back gracefully
    (feature the natural top item) rather than render a hero mid-grid.

    When found: the hero is removed from its original position and placed
    at index 0; ``diversify=True`` re-applies ``_reorder_diverse`` to the
    *remainder* only (removing the hero can re-introduce a same-domain
    adjacency that used to be broken up by the hero sitting between two
    same-domain items) — the hero itself is never subject to reordering.
    The remainder's diversification also avoids the hero's own domain for
    its first pick when possible, so the card immediately following the
    pinned hero doesn't land back-to-back with it. ``diversify=False`` (the
    ``"latest"`` sort) leaves the remainder's order untouched, preserving
    strict time order for every card after the pinned hero.
    """
    if hero_id is None:
        return None
    index = next((i for i, it in enumerate(items) if it.id == hero_id), None)
    if index is None:
        return None
    hero = items[index]
    rest = items[:index] + items[index + 1 :]
    if diversify:
        rest = _reorder_diverse(rest, avoid_domain=_domain_of(hero.source_url))
    return [hero, *rest]


def pick_onpage_hero_within_window(
    items: list, *, now: datetime
) -> Optional[uuid.UUID]:
    """Highest-``feed_score`` page item whose recency is within ``HERO_WINDOW``.

    Recovery for ``select_hero``'s global 48h pick landing *off* page 1 (Codex
    P2): when a full page of higher-``feed_score`` older items outranks the best
    recent item, that recent hero isn't among the rendered rows, so ``pin_hero``
    returns ``None``. Silently featuring the natural top item then buries the
    hero window entirely — an old, merely top-ranked story leads the magazine.

    Injecting the off-page hero is not an option: the recommended sort paginates
    by keyset on ``(feed_score, sort_at, id)``, and the hero sits *below* this
    page's cursor boundary, so it would reappear as a duplicate on a later page.
    Instead we surface the freshest high-scoring item the reader can actually see
    on this page. Returns ``None`` when no page item falls within the window (the
    caller then falls back to the natural top item as before).

    Recency uses each item's ``sort_at`` (``coalesce(published_at, created_at)``)
    — the same expression the feed sorts and ``select_hero``'s window filter by.
    ``feed_score`` may be ``None`` on some rows; a scored in-window item always
    beats an unscored one, mirroring the feed's ``DESC NULLS LAST`` order.
    """
    cutoff = now - HERO_WINDOW
    in_window = [it for it in items if it.sort_at >= cutoff]
    if not in_window:
        return None
    best = max(
        in_window,
        key=lambda it: (
            it.feed_score is not None,
            it.feed_score if it.feed_score is not None else 0.0,
            it.sort_at,
            it.id,
        ),
    )
    return best.id


# ------------------------------------------------------------------ #
# Query
# ------------------------------------------------------------------ #

def _doc_sort_expr():
    return func.coalesce(TechItem.published_at, TechItem.created_at)


def _entries_subquery():
    """One row per feed entry (ARG-243): ``entry_id, kind, sort_at,
    feed_score, rep_id, doc_count``.

    Two arms, ``UNION ALL``-ed:

    * **events** — every *live* event (``merged_into_id IS NULL``) with at
      least one evidence document. ``sort_at`` = newest evidence time,
      ``feed_score`` = max evidence score (NULL only when none is scored),
      ``rep_id`` = earliest-reported evidence (time ASC, id ASC).
    * **lone documents** — every document no *live* event claims, as a
      one-document entry whose ``rep_id`` is itself. A document linked only to
      a tombstoned event lands here too: moving links to the survivor is the
      merge-writer's job, and until it does, the document must not vanish from
      the feed.

    A document that belongs to two live events appears in both (decision 5).
    Event ids and document ids are both UUIDv4, so they share one keyset
    ``(…, entry_id)`` tiebreak without colliding in practice.
    """
    doc_sort = _doc_sort_expr()
    events = (
        select(
            EventDocument.event_id.label("entry_id"),
            literal("event", String).label("kind"),
            func.max(doc_sort).label("sort_at"),
            func.max(TechItem.feed_score).label("feed_score"),
            func.array_agg(
                aggregate_order_by(TechItem.id, doc_sort.asc(), TechItem.id.asc())
            )[1].label("rep_id"),
            func.count(TechItem.id).label("doc_count"),
        )
        .join(TechItem, TechItem.id == EventDocument.tech_item_id)
        .join(TechEvent, TechEvent.id == EventDocument.event_id)
        .where(TechEvent.merged_into_id.is_(None))
        .group_by(EventDocument.event_id)
    )
    claimed = (
        select(literal(1))
        .select_from(EventDocument)
        .join(TechEvent, TechEvent.id == EventDocument.event_id)
        .where(
            EventDocument.tech_item_id == TechItem.id,
            TechEvent.merged_into_id.is_(None),
        )
    )
    lone = select(
        TechItem.id.label("entry_id"),
        literal("item", String).label("kind"),
        doc_sort.label("sort_at"),
        TechItem.feed_score.label("feed_score"),
        TechItem.id.label("rep_id"),
        literal(1, Integer).label("doc_count"),
    ).where(~exists(claimed))
    return union_all(events, lone).subquery("feed_entries")


def _validate_category(category: Optional[str]) -> None:
    if category is not None and category not in ("Mainstream", "Alpha"):
        raise ValueError(f"invalid category: {category!r}")


def _entry_select(entries, *columns):
    """``SELECT columns FROM entries JOIN rep`` — the representative document
    is what category filtering (and every document-level field) keys off."""
    rep = TechItem
    return select(*columns).select_from(entries).join(
        rep, rep.id == entries.c.rep_id
    )


async def _fetch_source_domains(
    session: AsyncSession, event_ids: list[uuid.UUID]
) -> tuple[dict[uuid.UUID, tuple[str, ...]], dict[uuid.UUID, str]]:
    """Per event: distinct evidence domains (earliest report first), plus the
    first evidence cover image — used when the representative has none."""
    if not event_ids:
        return {}, {}
    doc_sort = _doc_sort_expr()
    rows = (
        await session.execute(
            select(
                EventDocument.event_id,
                TechItem.source_url,
                TechItem.image_url,
            )
            .join(TechItem, TechItem.id == EventDocument.tech_item_id)
            .where(EventDocument.event_id.in_(event_ids))
            .order_by(EventDocument.event_id, doc_sort.asc(), TechItem.id.asc())
        )
    ).all()
    domains: dict[uuid.UUID, list[str]] = defaultdict(list)
    images: dict[uuid.UUID, str] = {}
    for row in rows:
        domain = _domain_of(row.source_url)
        if domain and domain not in domains[row.event_id]:
            domains[row.event_id].append(domain)
        if row.image_url and row.event_id not in images:
            images[row.event_id] = row.image_url
    return {k: tuple(v) for k, v in domains.items()}, images


async def _hydrate(session: AsyncSession, rows) -> list[FeedItem]:
    event_ids = [row.entry_id for row in rows if row.kind == "event"]
    domains, images = await _fetch_source_domains(session, event_ids)
    items = []
    for row in rows:
        is_event = row.kind == "event"
        items.append(
            FeedItem(
                id=row.entry_id,
                # An unnamed event (naming not run yet / failed) borrows its
                # representative's headline so the card is never blank.
                title=(row.event_title if is_event and row.event_title else row.title),
                source_url=row.source_url,
                category=row.category,
                image_url=row.image_url or (images.get(row.entry_id) if is_event else None),
                summary=(
                    row.event_summary if is_event and row.event_summary else row.summary
                ),
                status=row.status,
                trust_score=row.trust_score,
                sort_at=row.sort_at,
                feed_score=row.feed_score,
                kind=row.kind,
                rep_id=row.rep_id,
                doc_count=row.doc_count,
                source_domains=domains.get(row.entry_id, ()) if is_event else (),
            )
        )
    return items


def _entry_columns(entries):
    return (
        entries.c.entry_id,
        entries.c.kind,
        entries.c.sort_at,
        entries.c.feed_score,
        entries.c.rep_id,
        entries.c.doc_count,
        TechItem.title,
        TechItem.source_url,
        TechItem.category,
        TechItem.image_url,
        TechItem.summary,
        TechItem.trust_score,
        TechEvent.title.label("event_title"),
        TechEvent.summary.label("event_summary"),
        UserAsset.status,
    )


def _entry_page_select(entries):
    return (
        _entry_select(entries, *_entry_columns(entries))
        .join(TechEvent, TechEvent.id == entries.c.entry_id, isouter=True)
        .join(UserAsset, UserAsset.tech_id == entries.c.rep_id, isouter=True)
    )


async def fetch_feed(
    session: AsyncSession,
    *,
    category: Optional[Category] = None,
    cursor: Optional[str] = None,
    limit: int = PAGE_SIZE,
    sort: FeedSort = "recommended",
) -> FeedPage:
    """Return one paginated page of feed entries (ARG-243: events).

    ``sort="recommended"`` (default, ARG-213) orders by the entry's
    ``feed_score`` descending with NULLs last, breaking ties by recency then
    entry id — so the NULL tail reads newest-first instead of in arbitrary
    UUID order — then applies a page-local same-domain-not-consecutive reorder
    before returning. ``sort="latest"`` orders strictly by the entry's newest
    evidence time, unreordered — the ARG-203 polling contract depends on this
    path staying strictly time-ordered.

    The Slack briefing column is intentionally not referenced here —
    that column is exclusively owned by the briefing pipeline.
    """
    if sort not in ("recommended", "latest"):
        raise ValueError(f"invalid feed sort: {sort!r}")
    _validate_category(category)

    entries = _entries_subquery()
    score = entries.c.feed_score
    sort_at = entries.c.sort_at
    entry_id = entries.c.entry_id

    stmt = _entry_page_select(entries).limit(limit + 1)

    if sort == "recommended":
        stmt = stmt.order_by(score.desc().nullslast(), sort_at.desc(), entry_id.desc())
    else:
        stmt = stmt.order_by(sort_at.desc(), entry_id.desc())

    if category is not None:
        stmt = stmt.where(TechItem.category == CategoryType(category))

    if cursor is not None:
        if sort == "recommended":
            cur_score, cur_sort, cur_id = decode_score_cursor(cursor)
            if cur_score is None:
                # Cursor is already in the NULLS-LAST tail: only other
                # null-score entries can sort after it.
                stmt = stmt.where(
                    score.is_(None)
                    & ((sort_at < cur_sort) | ((sort_at == cur_sort) & (entry_id < cur_id)))
                )
            else:
                stmt = stmt.where(
                    score.is_(None)
                    | (score < cur_score)
                    | ((score == cur_score) & (sort_at < cur_sort))
                    | ((score == cur_score) & (sort_at == cur_sort) & (entry_id < cur_id))
                )
        else:
            cur_sort, cur_id = decode_cursor(cursor)
            stmt = stmt.where(
                (sort_at < cur_sort) | ((sort_at == cur_sort) & (entry_id < cur_id))
            )

    rows = (await session.execute(stmt)).all()
    items = await _hydrate(session, rows[:limit])

    # Cursor for the *next* page must key off the true DB-order last row on
    # THIS page — computed before the display-only diversity reorder below —
    # or reordering would skip/duplicate items across the page boundary.
    next_cursor = None
    if len(rows) > limit and items:
        last = items[-1]
        next_cursor = (
            encode_score_cursor(last.feed_score, last.sort_at, last.id)
            if sort == "recommended"
            else encode_cursor(last.sort_at, last.id)
        )

    if sort == "recommended":
        items = _reorder_diverse(items)

    return FeedPage(items=items, next_cursor=next_cursor)


async def fetch_feed_entry(
    session: AsyncSession, entry_id: uuid.UUID
) -> Optional[FeedItem]:
    """One feed entry by id, for re-rendering a single card after Keep/Pass
    (ARG-243). ``None`` when no live entry carries that id anymore."""
    entries = _entries_subquery()
    stmt = _entry_page_select(entries).where(entries.c.entry_id == entry_id).limit(1)
    rows = (await session.execute(stmt)).all()
    if not rows:
        return None
    return (await _hydrate(session, rows))[0]


async def select_hero(
    session: AsyncSession, *, category: Optional[Category] = None
) -> Optional[uuid.UUID]:
    """The recommendation feed's magazine hero (ARG-213; entries since ARG-243).

    The highest-``feed_score`` entry whose recency (newest evidence time — the
    same ``sort_at`` the feed sorts by) falls within the last ``HERO_WINDOW``
    (48h); falls back to the highest-``feed_score`` entry overall when nothing
    qualifies within that window; ``None`` when no entry has a score at all.
    """
    _validate_category(category)

    entries = _entries_subquery()
    cutoff = datetime.now(timezone.utc) - HERO_WINDOW

    def _best(*conds):
        stmt = (
            _entry_select(entries, entries.c.entry_id)
            .where(entries.c.feed_score.is_not(None), *conds)
            .order_by(entries.c.feed_score.desc(), entries.c.entry_id.desc())
            .limit(1)
        )
        if category is not None:
            stmt = stmt.where(TechItem.category == CategoryType(category))
        return stmt

    row = (await session.execute(_best(entries.c.sort_at >= cutoff))).first()
    if row is not None:
        return row[0]
    row = (await session.execute(_best())).first()
    return row[0] if row is not None else None


async def latest_feed_cursor(
    session: AsyncSession, *, category: Optional[Category] = None
) -> Optional[str]:
    """The true global-latest entry's time-based cursor (review fix, ARG-213).

    Independent of whatever sort actually rendered the current page: under the
    "recommended" sort, page 1's newest card can be older than the genuinely
    newest entry, which would make ``count_new_since`` report a pre-existing
    entry as "new". One dedicated ``ORDER BY sort_at DESC, entry_id DESC
    LIMIT 1`` keeps the poll baseline honest.

    Returns ``None`` when there are no entries (after the category filter).
    """
    _validate_category(category)

    entries = _entries_subquery()
    stmt = (
        _entry_select(entries, entries.c.entry_id, entries.c.sort_at)
        .order_by(entries.c.sort_at.desc(), entries.c.entry_id.desc())
        .limit(1)
    )
    if category is not None:
        stmt = stmt.where(TechItem.category == CategoryType(category))

    row = (await session.execute(stmt)).first()
    if row is None:
        return None
    return encode_cursor(row.sort_at, row.entry_id)


async def count_new_since(
    session: AsyncSession,
    *,
    category: Optional[Category] = None,
    cursor: str,
) -> int:
    """Count feed entries newer than ``cursor`` (ARG-203 polling endpoint).

    Mirrors the ``latest`` sort's ordering rule, inverted: an entry is "new"
    when it sorts *before* the cursor position. Because an event's ``sort_at``
    is its newest evidence time, an existing event that just gained a document
    counts too (ARG-243 decision 2). ``decode_cursor`` raises ``ValueError`` on
    a malformed token — that propagates so the route can answer 400.

    Stays latest-based regardless of which sort the feed is rendering.
    """
    cur_sort, cur_id = decode_cursor(cursor)
    _validate_category(category)

    entries = _entries_subquery()
    sort_at = entries.c.sort_at
    stmt = _entry_select(entries, func.count()).where(
        (sort_at > cur_sort) | ((sort_at == cur_sort) & (entries.c.entry_id > cur_id))
    )
    if category is not None:
        stmt = stmt.where(TechItem.category == CategoryType(category))

    return (await session.execute(stmt)).scalar_one()
