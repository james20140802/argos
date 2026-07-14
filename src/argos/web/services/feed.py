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

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from argos.models.tech_item import CategoryType, TechItem
from argos.models.user_asset import AssetStatus, UserAsset


PAGE_SIZE: int = 20

# ARG-213: the hero is the highest-feed_score item published/created within
# this trailing window; see ``select_hero``.
HERO_WINDOW: timedelta = timedelta(hours=48)


Category = Literal["Mainstream", "Alpha"]
FeedSort = Literal["recommended", "latest"]


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


def _reorder_diverse(items: list, *, avoid_domain: Optional[str] = None) -> list:
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

async def fetch_feed(
    session: AsyncSession,
    *,
    category: Optional[Category] = None,
    cursor: Optional[str] = None,
    limit: int = PAGE_SIZE,
    sort: FeedSort = "recommended",
) -> FeedPage:
    """Return one paginated page of feed items.

    ``sort="recommended"`` (default, ARG-213) orders by ``feed_score``
    descending with NULLs last, breaking ties by recency then id — so the
    NULL tail (all rows right after the feed_score migration, and items added
    between scheduled rescores) reads newest-first instead of in arbitrary
    UUID order — then applies a page-local same-domain-not-consecutive reorder
    before returning. ``sort="latest"`` preserves the
    original ``coalesce(published_at, created_at)`` time order exactly as
    before, unreordered — the ARG-203 polling contract depends on this path
    staying strictly time-ordered.

    The Slack briefing column is intentionally not referenced here —
    that column is exclusively owned by the briefing pipeline.
    """
    if sort not in ("recommended", "latest"):
        raise ValueError(f"invalid feed sort: {sort!r}")

    sort_expr = func.coalesce(TechItem.published_at, TechItem.created_at)

    stmt = (
        select(
            TechItem.id,
            TechItem.title,
            TechItem.source_url,
            TechItem.category,
            TechItem.image_url,
            TechItem.summary,
            TechItem.trust_score,
            TechItem.feed_score,
            UserAsset.status,
            sort_expr.label("sort_at"),
        )
        .join(UserAsset, UserAsset.tech_id == TechItem.id, isouter=True)
        .limit(limit + 1)
    )

    if sort == "recommended":
        # feed_score DESC (NULLs last), then recency, then id. The recency
        # tiebreak is what gives the NULL tail — all rows right after the
        # migration, and items added between scheduled rescores — a sane
        # newest-first order instead of arbitrary UUID order.
        stmt = stmt.order_by(
            TechItem.feed_score.desc().nullslast(),
            sort_expr.desc(),
            TechItem.id.desc(),
        )
    else:
        stmt = stmt.order_by(sort_expr.desc(), TechItem.id.desc())

    if category is not None:
        if category not in ("Mainstream", "Alpha"):
            raise ValueError(f"invalid category: {category!r}")
        stmt = stmt.where(TechItem.category == CategoryType(category))

    if cursor is not None:
        if sort == "recommended":
            cur_score, cur_sort, cur_id = decode_score_cursor(cursor)
            if cur_score is None:
                # Cursor is already in the NULLS-LAST tail: only other
                # null-score rows can sort after it, keyed by recency
                # (sort_at desc) then id desc.
                stmt = stmt.where(
                    TechItem.feed_score.is_(None)
                    & (
                        (sort_expr < cur_sort)
                        | ((sort_expr == cur_sort) & (TechItem.id < cur_id))
                    )
                )
            else:
                stmt = stmt.where(
                    TechItem.feed_score.is_(None)
                    | (TechItem.feed_score < cur_score)
                    | ((TechItem.feed_score == cur_score) & (sort_expr < cur_sort))
                    | (
                        (TechItem.feed_score == cur_score)
                        & (sort_expr == cur_sort)
                        & (TechItem.id < cur_id)
                    )
                )
        else:
            cur_sort, cur_id = decode_cursor(cursor)
            stmt = stmt.where(
                (sort_expr < cur_sort)
                | ((sort_expr == cur_sort) & (TechItem.id < cur_id))
            )

    result = await session.execute(stmt)
    rows = result.all()

    items = [
        FeedItem(
            id=row.id,
            title=row.title,
            source_url=row.source_url,
            category=row.category,
            image_url=row.image_url,
            summary=row.summary,
            status=row.status,
            trust_score=row.trust_score,
            sort_at=row.sort_at,
            feed_score=row.feed_score,
        )
        for row in rows[:limit]
    ]

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


async def select_hero(
    session: AsyncSession, *, category: Optional[Category] = None
) -> Optional[uuid.UUID]:
    """The recommendation feed's magazine hero (ARG-213).

    The highest-``feed_score`` item whose recency —
    ``coalesce(published_at, created_at)`` — falls within the last
    ``HERO_WINDOW`` (48h); falls back to the highest-``feed_score`` item
    overall when nothing qualifies within that window; ``None`` when no item
    has a ``feed_score`` at all.

    The window uses the same ``coalesce(published_at, created_at)`` expression
    the feed sorts by, not bare ``created_at``: otherwise a months-old article
    (old ``published_at``) that was just crawled/added (recent ``created_at``)
    would be featured as the "recent" hero even though ``/feed`` orders it as
    old (Codex review).
    """
    if category is not None and category not in ("Mainstream", "Alpha"):
        raise ValueError(f"invalid category: {category!r}")

    cutoff = datetime.now(timezone.utc) - HERO_WINDOW
    recency_expr = func.coalesce(TechItem.published_at, TechItem.created_at)

    stmt = (
        select(TechItem.id)
        .where(TechItem.feed_score.is_not(None))
        .where(recency_expr >= cutoff)
        .order_by(TechItem.feed_score.desc(), TechItem.id.desc())
        .limit(1)
    )
    if category is not None:
        stmt = stmt.where(TechItem.category == CategoryType(category))

    row = (await session.execute(stmt)).first()
    if row is not None:
        return row[0]

    fallback_stmt = (
        select(TechItem.id)
        .where(TechItem.feed_score.is_not(None))
        .order_by(TechItem.feed_score.desc(), TechItem.id.desc())
        .limit(1)
    )
    if category is not None:
        fallback_stmt = fallback_stmt.where(TechItem.category == CategoryType(category))

    row = (await session.execute(fallback_stmt)).first()
    return row[0] if row is not None else None


async def latest_feed_cursor(
    session: AsyncSession, *, category: Optional[Category] = None
) -> Optional[str]:
    """The true global-latest item's time-based cursor (review fix, ARG-213).

    Independent of whatever sort actually rendered the current page. Before
    this helper existed, ``argos.web.app._render_feed`` derived the ARG-203
    poll baseline from ``max(sort_at, id)`` across the *rendered* page —
    under the "recommended" default sort, page 1 is ordered by
    ``feed_score``, so its max-``sort_at`` item can be older than the
    genuinely newest item in the table (which may have a low ``feed_score``
    and simply not appear on page 1 at all). That made ``count_new_since``
    treat a pre-existing, never-arrived item as "new" — a false "새 항목
    N개" pill. This runs one dedicated ``ORDER BY sort_at DESC, id DESC
    LIMIT 1`` query so the poll baseline is always the true newest item by
    wall-clock time, regardless of the active sort.

    Returns ``None`` when the table (after the optional category filter) is
    empty.
    """
    if category is not None and category not in ("Mainstream", "Alpha"):
        raise ValueError(f"invalid category: {category!r}")

    sort_expr = func.coalesce(TechItem.published_at, TechItem.created_at)
    stmt = (
        select(TechItem.id, sort_expr.label("sort_at"))
        .order_by(sort_expr.desc(), TechItem.id.desc())
        .limit(1)
    )
    if category is not None:
        stmt = stmt.where(TechItem.category == CategoryType(category))

    row = (await session.execute(stmt)).first()
    if row is None:
        return None
    return encode_cursor(row.sort_at, row.id)


async def count_new_since(
    session: AsyncSession,
    *,
    category: Optional[Category] = None,
    cursor: str,
) -> int:
    """Count feed items newer than ``cursor`` (ARG-203 polling endpoint).

    Mirrors the ``latest`` sort's ordering rule (``sort_expr`` desc, ``id``
    desc) but inverted: an item is "new" when it sorts *after* the cursor
    position, i.e. ``sort_expr > cur_sort`` or a tie broken by a greater id.
    ``decode_cursor`` raises ``ValueError`` on a malformed token — that
    propagates so the route can translate it into a 400.

    Deliberately unchanged by ARG-213: this stays latest-based regardless of
    which sort the feed itself is currently rendering in — see
    ``argos.web.app._render_feed``'s ``latest_cursor`` computation, which
    keeps feeding this a genuine time-based cursor even on the recommended
    page.
    """
    cur_sort, cur_id = decode_cursor(cursor)
    sort_expr = func.coalesce(TechItem.published_at, TechItem.created_at)

    stmt = select(func.count()).select_from(TechItem).where(
        (sort_expr > cur_sort) | ((sort_expr == cur_sort) & (TechItem.id > cur_id))
    )

    if category is not None:
        if category not in ("Mainstream", "Alpha"):
            raise ValueError(f"invalid category: {category!r}")
        stmt = stmt.where(TechItem.category == CategoryType(category))

    return (await session.execute(stmt)).scalar_one()
