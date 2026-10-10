"""원문 URL의 등록 도메인 — 같은 출처 보정의 키 (ARG-284 / ARG-295).

공개 접미사 목록(publicsuffix) 라이브러리를 쓰지 않는 이유: 새 의존성이 생기고,
목록을 최신으로 두려면 실행 중 내려받아야 하는데 둘 다 금지다. 대신 자주 나오는
이중 접미사만 내장한다. 근사이므로 한계가 있다 — github.com, arxiv.org,
huggingface.co 같은 플랫폼 도메인에서는 서로 다른 작성자의 글도 같은 출처로
친다. 접미사 표에 없는 호스팅 플랫폼(*.github.io, *.substack.com,
*.medium.com, *.blogspot.com 등)도 마찬가지로 서로 다른 작성자를 한 출처로
뭉친다. 그러므로 same_source_penalty 노브를 켜기 전에 이 표(또는
event_eval.PLATFORM_DOMAINS의 플랫폼 지표)를 다시 살펴야 한다. 그 영향은
Task 3(ARG-296)이 따로 잰다.
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit

_MULTI_PART_SUFFIXES = frozenset(
    {
        "co.uk", "org.uk", "ac.uk", "gov.uk",
        "co.kr", "or.kr", "go.kr", "ac.kr", "ne.kr",
        "co.jp", "ne.jp", "or.jp", "ac.jp",
        "com.au", "net.au", "org.au",
        "com.cn", "com.br", "com.tw", "co.in", "co.nz",
    }
)  # fmt: skip


def _hostname(url: str) -> str | None:
    try:
        host = urlsplit(url).hostname
        if not host and "//" not in url:
            # 스킴 없는 문자열("openai.com/x")은 경로로 파싱되므로 다시 읽는다.
            host = urlsplit("//" + url).hostname
    except ValueError:
        return None
    return host or None


def registered_domain(url: str | None) -> str | None:
    """URL의 등록 도메인(소문자). 알 수 없으면 ``None`` — 같은 출처로 치지 않는다."""
    if not url or not url.strip():
        return None
    host = _hostname(url.strip())
    if not host:
        return None
    host = host.lower().rstrip(".")
    if not host or any(ch.isspace() for ch in host):
        return None
    try:
        ipaddress.ip_address(host)
        return host
    except ValueError:
        pass
    labels = host.split(".")
    if len(labels) == 1:
        return host
    take = 3 if ".".join(labels[-2:]) in _MULTI_PART_SUFFIXES else 2
    if len(labels) <= take:
        return host
    return ".".join(labels[-take:])
