"""애매한 짝에 "같은 발표인가"를 로컬 LLM으로 묻는다 — ARG-307.

재군집이 매긴 짝 점수가 경계(τ 근처)에 있고 두 글이 같은 사이트에서 나왔다면,
점수만으로는 "같은 발표의 두 글"과 "같은 회사의 다른 발표"를 가르기 어렵다
(예: Intuit×OpenAI 협업과 Target×OpenAI 협업). 그 자리에서만 LLM에게 묻는다.

**LLM은 간선을 지우기만 한다.** "다른 발표"라는 답이 나온 짝은 그래프에서 빠지고,
"같은 발표"라는 답은 아무것도 바꾸지 않는다. 점수 함수는 그대로라 낮·밤이 같은
점수를 쓴다는 원칙(ARG-282)은 유지된다. 이 모듈은 판정 코어(`recluster_core`)
밖에 있어서, 코어는 계속 순수·동기다.

이 파일은 세 덩어리다: 판단 캐시(같은 짝을 두 번 묻지 않는다), 순수 함수(짝 고르기·
프롬프트·응답 읽기), 비동기 오케스트레이션.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

if TYPE_CHECKING:
    from argos.brain.recluster_core import WeightedEdge

logger = logging.getLogger(__name__)

PROMPT_VERSION = 1
"""프롬프트를 바꾸면 올린다 — 캐시 키에 들어가 옛 답을 자연히 무효화한다."""

SAME = "SAME"
DIFFERENT = "DIFFERENT"


@dataclass(frozen=True)
class CachedVerdict:
    verdict: str
    reason: str
    asked_at: str


_CacheKey = tuple[str, str, str, int, str]


def _key(
    left: uuid.UUID, right: uuid.UUID, model: str, prompt_version: int, input_sha: str
) -> _CacheKey:
    low, high = sorted([left, right])
    return (str(low), str(high), model, prompt_version, input_sha)


class PairVerdictCache:
    """JSONL 파일 캐시. 키 = (두 id 정렬, 모델, 프롬프트 버전, 입력 해시).

    입력 해시는 프롬프트에 들어간 제목·요약의 해시라, 본문이 보강돼 바뀌면 다시
    묻는다. 쓰기는 임시 파일 + `os.replace`라 중간에 죽어도 옛 파일이 온전하다.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._entries: dict[_CacheKey, CachedVerdict] = {}
        self._dirty = False

    @classmethod
    def load(cls, path: Path) -> "PairVerdictCache":
        cache = cls(path)
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return cache
        except OSError as exc:
            logger.warning("pair_check cache unreadable (%s): %s", path, exc)
            return cache
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                key = (
                    str(row["l"]),
                    str(row["r"]),
                    str(row["model"]),
                    int(row["pv"]),
                    str(row["sha"]),
                )
                verdict = str(row["verdict"])
                if verdict not in (SAME, DIFFERENT):
                    raise ValueError(verdict)
                cache._entries[key] = CachedVerdict(
                    verdict=verdict,
                    reason=str(row.get("reason", "")),
                    asked_at=str(row.get("asked_at", "")),
                )
            except (ValueError, KeyError, TypeError):
                logger.warning(
                    "pair_check cache %s line %d is corrupt; skipped", path, number
                )
        return cache

    def __len__(self) -> int:
        return len(self._entries)

    def get(
        self,
        left: uuid.UUID,
        right: uuid.UUID,
        *,
        model: str,
        input_sha: str,
        prompt_version: int = PROMPT_VERSION,
    ) -> CachedVerdict | None:
        return self._entries.get(_key(left, right, model, prompt_version, input_sha))

    def put(
        self,
        left: uuid.UUID,
        right: uuid.UUID,
        *,
        model: str,
        input_sha: str,
        verdict: str,
        reason: str,
        prompt_version: int = PROMPT_VERSION,
    ) -> None:
        self._entries[_key(left, right, model, prompt_version, input_sha)] = (
            CachedVerdict(
                verdict=verdict,
                reason=reason,
                asked_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
        )
        self._dirty = True

    def flush(self) -> None:
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            for (low, high, model, pv, sha), value in sorted(self._entries.items()):
                handle.write(
                    json.dumps(
                        {
                            "l": low,
                            "r": high,
                            "model": model,
                            "pv": pv,
                            "sha": sha,
                            "verdict": value.verdict,
                            "reason": value.reason,
                            "asked_at": value.asked_at,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        os.replace(tmp, self.path)
        self._dirty = False


def default_cache_path() -> Path:
    base = os.environ.get("XDG_STATE_HOME")
    root = Path(base) if base else Path.home() / ".local" / "state"
    return root / "argos" / "pair_checks.jsonl"


# --- 순수 함수: 짝 고르기 · 프롬프트 · 응답 읽기 ---------------------------------

SUMMARY_CHARS = 400

_PROMPT = """\
Two articles are shown. Decide whether they cover the SAME single announcement, \
release, or event (for example the announcement post and its system card, or two \
write-ups of the same launch), or DIFFERENT ones (for example the same company \
announcing two separate partnerships, products, or customer stories).

Examples:
- "Intuit and OpenAI partner on ..." vs "Target and OpenAI partner on ..." -> DIFFERENT
- "Introducing Codex" vs "o3 system card addendum: Codex" -> SAME

Article 1
Title: {title_1}
Summary: {summary_1}

Article 2
Title: {title_2}
Summary: {summary_2}

Answer with exactly one word, SAME or DIFFERENT, then a short one-line reason."""


def select_ambiguous_pairs(
    scored: "Sequence[WeightedEdge]",
    sources: "Mapping[uuid.UUID, str | None]",
    *,
    lower: float,
    upper: float,
    same_source_only: bool,
) -> "tuple[WeightedEdge, ...]":
    """점수가 `[lower, upper)`에 있는 짝만 고른다.

    `same_source_only`면 두 문서의 등록 도메인이 같은 짝만 남긴다. 출처를 모르는
    문서(None)는 같은 출처로 치지 않는다 — 모르는 것을 같다고 치면 근거 없이 묻는
    셈이다(ARG-306과 같은 원칙). 결과는 `(left_id, right_id)` 오름차순이라
    입력 순서와 무관하다.
    """
    picked = []
    for edge in scored:
        if not (lower <= edge.weight < upper):
            continue
        if same_source_only:
            left = sources.get(edge.left_id)
            right = sources.get(edge.right_id)
            if left is None or right is None or left != right:
                continue
        picked.append(edge)
    return tuple(sorted(picked, key=lambda edge: (edge.left_id, edge.right_id)))


def _squash(text: str | None) -> str:
    return " ".join((text or "").split())


def _clip(summary: str | None) -> str:
    return _squash(summary)[:SUMMARY_CHARS]


def build_prompt(
    first: "tuple[str, str | None]", second: "tuple[str, str | None]"
) -> str:
    """각각 (제목, 요약). 요약은 `SUMMARY_CHARS`자로 자른다."""
    return _PROMPT.format(
        title_1=_squash(first[0]),
        summary_1=_clip(first[1]),
        title_2=_squash(second[0]),
        summary_2=_clip(second[1]),
    )


def input_sha(first: "tuple[str, str | None]", second: "tuple[str, str | None]") -> str:
    """프롬프트에 들어가는 텍스트의 해시. 순서와 무관하다."""
    parts = sorted(
        f"{_squash(title)}\x1f{_clip(summary)}" for title, summary in (first, second)
    )
    return hashlib.sha1("\x1e".join(parts).encode("utf-8")).hexdigest()


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_VERDICT_WORD = re.compile(r"\b(SAME|DIFFERENT)\b", re.IGNORECASE)


def parse_verdict(text: str) -> "tuple[str, str] | None":
    """`(SAME|DIFFERENT, 근거)`. 판단 단어가 없거나 둘 다 있으면 None(= 판단 없음)."""
    body = _THINK_BLOCK.sub("", text or "").strip()
    found = {match.group(1).upper() for match in _VERDICT_WORD.finditer(body)}
    if len(found) != 1:
        return None
    first = _VERDICT_WORD.search(body)
    assert first is not None
    rest = body[first.end():].strip()
    reason = rest.splitlines()[0] if rest else ""
    return found.pop(), reason.lstrip(" -:–—").strip()


# --- 비동기 오케스트레이션 ---------------------------------------------------------


@dataclass(frozen=True)
class PairVerdict:
    """한 짝에 대한 판단 한 건. `cached`면 이번 실행에서 묻지 않고 캐시에서 읽었다."""

    left_id: uuid.UUID
    right_id: uuid.UUID
    left_title: str
    right_title: str
    verdict: str
    reason: str
    cached: bool


@dataclass(frozen=True)
class PairCheckReport:
    """한 번의 확인 결과. CLI 미리보기가 그대로 찍는다."""

    model: str
    vetoed: frozenset[tuple[uuid.UUID, uuid.UUID]] = frozenset()
    band_pairs: int = 0
    asked: int = 0
    cache_hits: int = 0
    skipped: int = 0
    skipped_reason: str | None = None
    elapsed_s: float = 0.0
    verdicts: tuple[PairVerdict, ...] = ()


async def _ask_ollama(model: str, prompt: str) -> str:
    from argos.brain.ollama_client import query_ollama

    # temperature 0 + think 끔: 짧은 판정이라 추론 토큰이 필요 없다. keep_alive는
    # 배치 안에서만 살려 두고, 끝나면 호출부가 내린다.
    return await query_ollama(
        model, prompt, keep_alive="2m", temperature=0.0, think=False
    )


async def _unload_ollama(model: str) -> None:
    from argos.brain.ollama_client import unload_model

    await unload_model(model)


async def fetch_pair_texts(session, ids) -> "dict[uuid.UUID, tuple[str, str | None]]":
    """밴드 짝에 걸린 문서만 제목·요약을 읽는다(읽기 전용)."""
    from sqlalchemy import select

    from argos.models.tech_item import TechItem

    id_list = sorted(set(ids))
    if not id_list:
        return {}
    rows = await session.execute(
        select(TechItem.id, TechItem.title, TechItem.summary).where(
            TechItem.id.in_(id_list)
        )
    )
    return {row.id: (row.title or "", row.summary) for row in rows}


async def check_ambiguous_pairs(
    session,
    documents: "Sequence",
    scored_pairs: "Sequence[WeightedEdge]",
    *,
    config,
    cache: PairVerdictCache,
    ask=None,
    load_texts=None,
    max_calls: int | None = None,
    unload=None,
) -> PairCheckReport:
    """밴드 짝을 골라 캐시를 먼저 보고, 미스만 LLM에 묻는다.

    Ollama가 안 닿거나 응답이 이상해도 예외를 올리지 않는다 — 교정은 이 확인이 꺼진
    것과 같은 결과로 계속 진행돼야 한다(캐시에 이미 있는 거부는 그대로 적용).
    `ask(model, prompt) -> str`, `load_texts(session, ids)`, `unload(model)`은
    테스트가 갈아 끼우는 자리다.
    """
    started = time.monotonic()
    model = config.pair_check_model
    ask = ask or _ask_ollama
    load_texts = load_texts or fetch_pair_texts
    unload = unload or _unload_ollama
    limit = config.pair_check_max_calls if max_calls is None else max_calls

    band = select_ambiguous_pairs(
        scored_pairs,
        {doc.tech_item_id: doc.features.source for doc in documents},
        lower=config.join_threshold,
        upper=config.pair_check_upper,
        same_source_only=config.pair_check_same_source_only,
    )
    if not band:
        return PairCheckReport(model=model, elapsed_s=time.monotonic() - started)

    texts = await load_texts(
        session, {i for edge in band for i in (edge.left_id, edge.right_id)}
    )
    vetoed: set[tuple[uuid.UUID, uuid.UUID]] = set()
    verdicts: list[PairVerdict] = []
    asked = cache_hits = skipped = 0
    infra_down = False
    reasons: list[str] = []
    over_cap = 0

    def record(edge, verdict, reason, *, cached):
        left_text, right_text = texts[edge.left_id], texts[edge.right_id]
        verdicts.append(
            PairVerdict(
                edge.left_id, edge.right_id, left_text[0], right_text[0],
                verdict, reason, cached,
            )
        )
        if verdict == DIFFERENT:
            vetoed.add((edge.left_id, edge.right_id))

    try:
        for edge in band:
            left_text, right_text = texts.get(edge.left_id), texts.get(edge.right_id)
            if left_text is None or right_text is None:
                skipped += 1
                continue
            sha = input_sha(left_text, right_text)
            hit = cache.get(edge.left_id, edge.right_id, model=model, input_sha=sha)
            if hit is not None:
                cache_hits += 1
                record(edge, hit.verdict, hit.reason, cached=True)
                continue
            if infra_down:
                skipped += 1
                continue
            if asked >= limit:
                skipped += 1
                over_cap += 1
                continue
            asked += 1
            try:
                answer = await ask(model, build_prompt(left_text, right_text))
            except Exception as exc:  # noqa: BLE001 — 교정을 막지 않는다
                from argos.brain.ollama_client import OllamaInfraError

                infra_down = True
                asked -= 1  # 답을 못 받은 시도는 물음으로 세지 않는다
                skipped += 1
                label = "Ollama 연결 실패" if isinstance(exc, OllamaInfraError) else (
                    f"LLM 호출 실패({type(exc).__name__})"
                )
                reasons.append(label)
                logger.warning("pair_check: %s — 남은 짝은 건너뛴다: %s", label, exc)
                continue
            parsed = parse_verdict(answer)
            if parsed is None:
                logger.info("pair_check: unparseable answer for %s/%s", edge.left_id, edge.right_id)
                continue
            verdict, reason = parsed
            cache.put(
                edge.left_id, edge.right_id,
                model=model, input_sha=sha, verdict=verdict, reason=reason,
            )
            record(edge, verdict, reason, cached=False)
    finally:
        try:
            cache.flush()
        except OSError as exc:
            logger.warning("pair_check: cache write failed: %s", exc)
        if asked:
            try:
                await unload(model)
            except Exception as exc:  # noqa: BLE001
                logger.warning("pair_check: unload %s failed: %s", model, exc)

    if over_cap:
        reasons.append(f"호출 상한 {limit} 초과 {over_cap}건 미확인")
    return PairCheckReport(
        model=model,
        vetoed=frozenset(vetoed),
        band_pairs=len(band),
        asked=asked,
        cache_hits=cache_hits,
        skipped=skipped,
        skipped_reason="; ".join(reasons) or None,
        elapsed_s=time.monotonic() - started,
        verdicts=tuple(verdicts),
    )
