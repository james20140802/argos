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
