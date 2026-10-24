"""사건 요약 입력 예산 -- ARG-310.

원칙: **메모리는 ``num_ctx``로 고정하고, 입력을 그 안에 맞춘다.** 기사가 많거나
본문이 긴 사건은 ``num_ctx``를 키우는 대신 청크로 나눠 맵리듀스로 접는다
(``event_naming.generate_event_naming``). 이 모듈은 DB도 LLM도 모르는 순수
함수만 둔다.

토큰 수는 토크나이저 없이 **문자 수 / chars_per_tok**으로 어림한다. 어림이
틀려 컨텍스트를 넘기면 Ollama가 앞부분을 조용히 버리므로, 호출부가
``prompt_eval_count``로 넘침을 감지해 경고한다.
"""

from __future__ import annotations

from typing import Sequence

BLOCK_SEPARATOR = "\n\n"
"""근거 블록을 이어 붙일 때 쓰는 구분자 -- 청크 크기 계산에 포함된다."""

MAX_FOLD_DEPTH = 3
"""맵리듀스를 접는 단계 상한. 넘기면 호출부가 실패(None)로 돌린다."""

_MIN_BUDGET_CHARS = 500
"""템플릿이 컨텍스트를 거의 다 먹는 비정상 설정에서도 0 이하가 되지 않게 하는 바닥."""


def budget_chars(
    *,
    num_ctx: int,
    output_reserve_tok: int,
    chars_per_tok: float,
    template_chars: int,
) -> int:
    """프롬프트 템플릿을 뺀, 근거 텍스트에 쓸 수 있는 문자 수 예산."""
    usable_tokens = max(num_ctx - output_reserve_tok, 0)
    return max(int(usable_tokens * chars_per_tok) - template_chars, _MIN_BUDGET_CHARS)


def joined_length(blocks: Sequence[str]) -> int:
    """``BLOCK_SEPARATOR``로 이어 붙였을 때의 길이."""
    if not blocks:
        return 0
    return sum(len(b) for b in blocks) + len(BLOCK_SEPARATOR) * (len(blocks) - 1)


def pack_chunks(blocks: Sequence[str], budget: int) -> list[list[str]]:
    """블록을 순서대로 ``budget`` 문자 이하 청크로 묶는다.

    순서를 보존하고(탐욕적 순차 채우기) 빈 블록은 버린다. 블록 하나가 예산보다
    크면 예산 크기로 잘라 단독 청크에 넣는다 -- 한 문서가 호출을 터뜨리지
    못하게 하는 최후 방어선이다(정상 경로에선 ``doc_max_chars``가 먼저 자른다).
    """
    chunks: list[list[str]] = []
    current: list[str] = []
    for block in blocks:
        if not block:
            continue
        if len(block) > budget:
            block = block[:budget]
        if current and joined_length([*current, block]) > budget:
            chunks.append(current)
            current = []
        current.append(block)
    if current:
        chunks.append(current)
    return chunks
