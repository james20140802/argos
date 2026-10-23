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

import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

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
