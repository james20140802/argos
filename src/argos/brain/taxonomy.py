"""역할 갈래 트리 — 순수 편집·검증 함수와 내장 시드 (ARG-311).

기술이 "무슨 역할인가"를 묻는 분류 틀(역할 축)이다. 위 갈래(3~6개) 아래에
아래 갈래가 달리는 2단 트리이고, 이 모듈은 메모리 위의 트리만 다룬다 — DB도
LLM도 건드리지 않는다. 확정본 저장은 ``taxonomy_store``, 초안 생성은
``taxonomy_draft``, 명령은 ``taxonomy_cli`` 소관이다.

편집 함수는 전부 **새 트리를 반환**하고(입력 불변), 결과가 ``validate``를
통과하지 못하면 ``TaxonomyError``로 거부한다. 노드의 ``id``는 확정본에서 온
노드만 갖고(새 노드는 ``None``), 편집이 이를 보존한다 — 이름 바꾸기가 ARG-313의
배정을 깨지 않게 하려는 장치다.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Sequence

MIN_TOP = 3
MAX_TOP = 6
MIN_NAME_LEN = 2
MAX_NAME_LEN = 40
MIN_DESCRIPTION_LEN = 10
MAX_EXAMPLES = 5

_VAGUE_TOP_NAMES = frozenset({"ai", "llm", "기타", "other", "others", "etc"})
_WORD_LIST_TOKEN = re.compile(r"^[a-z0-9][a-z0-9 .\-_]*$")


class TaxonomyError(ValueError):
    """트리가 규칙을 어기거나 편집 대상을 찾지 못했을 때."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


@dataclass(frozen=True)
class Branch:
    """갈래 하나. ``children``가 비어 있지 않으면 위 갈래다."""

    name: str
    description: str
    id: str | None = None
    children: tuple["Branch", ...] = ()
    examples: tuple[str, ...] = field(default=())


Tree = tuple[Branch, ...]


# --------------------------------------------------------------------- #
# 조회
# --------------------------------------------------------------------- #
def iter_branches(tree: Tree) -> Iterable[tuple[int, Branch, Branch | None]]:
    """(깊이, 갈래, 부모)를 깊이 우선으로. 위 갈래 깊이=1."""
    for top in tree:
        yield 1, top, None
        for child in top.children:
            yield 2, child, top


def find(tree: Tree, name: str) -> tuple[int, Branch, Branch | None]:
    for depth, branch, parent in iter_branches(tree):
        if branch.name == name:
            return depth, branch, parent
    raise TaxonomyError([f"갈래를 찾을 수 없습니다: {name!r}"])


# --------------------------------------------------------------------- #
# 검증
# --------------------------------------------------------------------- #
def looks_like_word_list(name: str) -> bool:
    """군집 키워드 나열("claude, went, seconds")처럼 보이는 이름인가."""
    if "," not in name:
        return False
    tokens = [t.strip() for t in name.split(",")]
    tokens = [t for t in tokens if t]
    return len(tokens) >= 3 and all(_WORD_LIST_TOKEN.match(t) for t in tokens)


def _name_errors(branch: Branch, label: str) -> list[str]:
    errors: list[str] = []
    name = branch.name.strip()
    if not (MIN_NAME_LEN <= len(name) <= MAX_NAME_LEN):
        errors.append(
            f"{label} 이름 길이는 {MIN_NAME_LEN}~{MAX_NAME_LEN}자여야 합니다: {branch.name!r}"
        )
    if looks_like_word_list(name):
        errors.append(f"{label} 이름이 단어 나열처럼 보입니다: {branch.name!r}")
    if len(branch.description.strip()) < MIN_DESCRIPTION_LEN:
        errors.append(
            f"{label} 설명은 {MIN_DESCRIPTION_LEN}자 이상이어야 합니다: {branch.name!r}"
        )
    return errors


def validate(tree: Tree) -> list[str]:
    """오류 목록(비어 있으면 통과). 경고는 ``warnings``."""
    errors: list[str] = []
    if not (MIN_TOP <= len(tree) <= MAX_TOP):
        errors.append(f"위 갈래는 {MIN_TOP}~{MAX_TOP}개여야 합니다 (현재 {len(tree)}개)")

    seen_names: set[str] = set()
    seen_ids: set[str] = set()
    for depth, branch, _parent in iter_branches(tree):
        label = "위 갈래" if depth == 1 else "아래 갈래"
        errors.extend(_name_errors(branch, label))
        if branch.name in seen_names:
            errors.append(f"갈래 이름이 중복됩니다: {branch.name!r}")
        seen_names.add(branch.name)
        if branch.id is not None:
            if branch.id in seen_ids:
                errors.append(f"갈래 id가 중복됩니다: {branch.id}")
            seen_ids.add(branch.id)
        if len(branch.examples) > MAX_EXAMPLES:
            errors.append(f"예시는 {MAX_EXAMPLES}개 이하여야 합니다: {branch.name!r}")
        if depth == 1 and not branch.children:
            errors.append(f"위 갈래에는 아래 갈래가 1개 이상 필요합니다: {branch.name!r}")
        if depth == 2 and branch.children:
            errors.append(f"깊이는 2단까지입니다: {branch.name!r} 아래에 더 둘 수 없습니다")
    return errors


def warnings(tree: Tree) -> list[str]:
    """오류는 아니지만 검토가 필요한 점."""
    out: list[str] = []
    for top in tree:
        if top.name.strip().lower() in _VAGUE_TOP_NAMES:
            out.append(f"위 갈래 이름이 너무 포괄적입니다: {top.name!r}")
    return out


def _checked(tree: Tree) -> Tree:
    errors = validate(tree)
    if errors:
        raise TaxonomyError(errors)
    return tree


# --------------------------------------------------------------------- #
# 편집 (모두 새 트리를 반환)
# --------------------------------------------------------------------- #
def _map_branch(tree: Tree, name: str, fn) -> Tree:
    """이름이 ``name``인 갈래에 ``fn``을 적용한 새 트리. 없으면 오류."""
    find(tree, name)
    out: list[Branch] = []
    for top in tree:
        if top.name == name:
            out.append(fn(top))
            continue
        children = tuple(fn(c) if c.name == name else c for c in top.children)
        out.append(replace(top, children=children))
    return tuple(out)


def rename(tree: Tree, name: str, new_name: str) -> Tree:
    new_name = new_name.strip()
    if new_name != name:
        for _, branch, _p in iter_branches(tree):
            if branch.name == new_name:
                raise TaxonomyError([f"이미 있는 이름입니다: {new_name!r}"])
    return _checked(_map_branch(tree, name, lambda b: replace(b, name=new_name)))


def redescribe(tree: Tree, name: str, description: str) -> Tree:
    return _checked(
        _map_branch(tree, name, lambda b: replace(b, description=description.strip()))
    )


def merge(tree: Tree, absorbed: str, into: str) -> Tree:
    """``absorbed``를 ``into``에 합친다. ``into``의 id·이름·설명이 남는다."""
    if absorbed == into:
        raise TaxonomyError(["같은 갈래끼리는 합칠 수 없습니다"])
    depth_a, a, _ = find(tree, absorbed)
    depth_b, _b, _ = find(tree, into)
    if depth_a != depth_b:
        raise TaxonomyError(["위 갈래는 위 갈래끼리, 아래 갈래는 아래 갈래끼리만 합칠 수 있습니다"])

    out: list[Branch] = []
    for top in tree:
        if depth_a == 1:
            if top.name == absorbed:
                continue
            if top.name == into:
                top = replace(top, children=top.children + a.children)
        else:
            top = replace(
                top, children=tuple(c for c in top.children if c.name != absorbed)
            )
        out.append(top)
    return _checked(tuple(out))


def split(
    tree: Tree,
    name: str,
    parts: Sequence[tuple[str, str]],
    moves: dict[str, str] | None = None,
) -> Tree:
    """갈래를 둘 이상으로 나눈다.

    아래 갈래: 같은 위치에 형제 ``parts``로 바뀐다(원본 제거).
    위 갈래: 자식을 전부 ``moves``(자식 이름 -> 새 위 갈래 이름)로 배치해야 한다.
    새 갈래의 id는 ``None``이다(자식의 id는 보존).
    """
    if len(parts) < 2:
        raise TaxonomyError(["나누려면 새 갈래가 2개 이상 필요합니다"])
    depth, target, _parent = find(tree, name)
    new_names = [p[0].strip() for p in parts]
    if len(set(new_names)) != len(new_names):
        raise TaxonomyError(["새 갈래 이름이 서로 겹칩니다"])
    for n in new_names:
        if n != name and any(b.name == n for _, b, _p in iter_branches(tree)):
            raise TaxonomyError([f"이미 있는 이름입니다: {n!r}"])

    new_branches = [Branch(name=n.strip(), description=d.strip()) for n, d in parts]
    if depth == 2:
        out: list[Branch] = []
        for top in tree:
            children: list[Branch] = []
            for c in top.children:
                if c.name == name:
                    children.extend(new_branches)
                else:
                    children.append(c)
            out.append(replace(top, children=tuple(children)))
        return _checked(tuple(out))

    moves = moves or {}
    child_names = {c.name for c in target.children}
    errors = [
        f"--move의 아래 갈래가 {name!r} 안에 없습니다: {k!r}"
        for k in moves
        if k not in child_names
    ]
    errors += [
        f"--move의 대상이 새 갈래가 아닙니다: {v!r}"
        for v in moves.values()
        if v not in new_names
    ]
    errors += [
        f"아래 갈래를 배치하지 않았습니다 (--move 필요): {c!r}"
        for c in sorted(child_names - set(moves))
    ]
    if errors:
        raise TaxonomyError(errors)
    buckets: dict[str, list[Branch]] = {n: [] for n in new_names}
    for c in target.children:
        buckets[moves[c.name]].append(c)
    out_tree: list[Branch] = []
    for top in tree:
        if top.name == name:
            for b in new_branches:
                out_tree.append(replace(b, children=tuple(buckets[b.name])))
        else:
            out_tree.append(top)
    return _checked(tuple(out_tree))


def delete(tree: Tree, name: str, *, force: bool = False) -> Tree:
    depth, target, _ = find(tree, name)
    if depth == 1 and target.children and not force:
        raise TaxonomyError(
            [f"{name!r} 아래에 아래 갈래 {len(target.children)}개가 있습니다 — 지우려면 --force"]
        )
    out: list[Branch] = []
    for top in tree:
        if top.name == name:
            continue
        out.append(replace(top, children=tuple(c for c in top.children if c.name != name)))
    return _checked(tuple(out))


# --------------------------------------------------------------------- #
# 직렬화
# --------------------------------------------------------------------- #
def branch_to_dict(b: Branch) -> dict:
    d: dict = {"id": b.id, "name": b.name, "description": b.description}
    if b.examples:
        d["examples"] = list(b.examples)
    if b.children:
        d["children"] = [branch_to_dict(c) for c in b.children]
    return d


def branch_from_dict(d: dict) -> Branch:
    if not isinstance(d, dict) or "name" not in d:
        raise TaxonomyError(["갈래 항목에 name이 없습니다"])
    return Branch(
        name=str(d["name"]),
        description=str(d.get("description") or ""),
        id=(str(d["id"]) if d.get("id") else None),
        children=tuple(branch_from_dict(c) for c in d.get("children") or []),
        examples=tuple(str(e) for e in d.get("examples") or []),
    )


def tree_to_dict(tree: Tree) -> dict:
    return {"tree": [branch_to_dict(b) for b in tree]}


def tree_from_dict(data: dict) -> Tree:
    if not isinstance(data, dict) or not isinstance(data.get("tree"), list):
        raise TaxonomyError(["파일에 'tree' 목록이 없습니다"])
    return tuple(branch_from_dict(b) for b in data["tree"])


def save_file(tree: Tree, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(tree_to_dict(tree), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_file(path: Path) -> Tree:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TaxonomyError([f"JSON을 읽을 수 없습니다: {exc}"]) from exc
    return tree_from_dict(data)


def render(tree: Tree) -> str:
    """사람이 읽는 들여쓰기 출력."""
    lines: list[str] = []
    for depth, b, _p in iter_branches(tree):
        indent = "" if depth == 1 else "    "
        bullet = "■" if depth == 1 else "-"
        lines.append(f"{indent}{bullet} {b.name}")
        lines.append(f"{indent}    {b.description}")
        if b.examples:
            lines.append(f"{indent}    예: {', '.join(b.examples)}")
    return "\n".join(lines)


# --------------------------------------------------------------------- #
# 내장 시드 (ARG-311 성공 장면 4 / D3)
# --------------------------------------------------------------------- #
def _leaf(name: str, description: str) -> Branch:
    return Branch(name=name, description=description)


SPIKE_SEED: Tree = (
    Branch(
        name="AI 모델 및 엔진",
        description="추론의 중심이 되는 모델 자체와 그것을 올려 돌리는 엔진.",
        children=(
            _leaf("기초 모델 (Foundation Models)", "범용 대형·프런티어·오픈 웨이트 모델."),
            _leaf("추론·특화 모델 (Reasoning Models)", "비전·음성 등 특정 모달리티나 추론 작업에 맞춘 모델."),
            _leaf("모델 실행·서빙 (Inference & Serving)", "모델을 배포하고 실행하는 엔진, 프레임워크, 로컬 러너."),
        ),
    ),
    Branch(
        name="AI 에이전트 및 어시스턴트",
        description="사람 대신 일을 수행하는 에이전트와 어시스턴트 제품.",
        children=(
            _leaf("코딩 에이전트 (Coding Agents)", "소프트웨어 개발과 코딩 작업을 자동화하는 도구와 에이전트."),
            _leaf("범용·개인용 에이전트 (General Agents)", "업무·개인 비서나 일반 워크플로를 맡는 AI 에이전트."),
        ),
    ),
    Branch(
        name="에이전트 인프라 및 도구",
        description="에이전트가 돌아가고 외부와 연결되도록 받쳐 주는 기반.",
        children=(
            _leaf("에이전트 오케스트레이션 (Orchestration)", "모델 실행, 도구 연동, 다중 에이전트 협업을 관리하는 프레임워크와 하니스."),
            _leaf("연결·게이트웨이 (Connectivity & Gateways)", "에이전트를 외부 도구·서버·모델 제공자와 잇는 프로토콜, API, 게이트웨이."),
        ),
    ),
    Branch(
        name="데이터 및 개발 인프라",
        description="데이터를 보관하고 코드를 관리·실행하는 일반 개발 기반.",
        children=(
            _leaf("데이터베이스·저장소 (Databases & Storage)", "데이터를 저장하고 질의하는 분석용·관계형·단일 파일 데이터베이스."),
            _leaf("버전 관리·협업 (Version Control)", "코드 변경 이력을 관리하고 함께 작업하게 해 주는 도구."),
            _leaf("런타임·배포 (Runtime & Deployment)", "애플리케이션을 실행하고 배포하는 서버리스·엣지 런타임과 플랫폼."),
        ),
    ),
    Branch(
        name="평가",
        description="모델과 에이전트의 성능을 재고 비교하는 방법과 도구.",
        children=(
            _leaf("벤치마크 (Benchmarks)", "모델과 에이전트의 성능을 평가하는 표준화된 시험과 지표."),
        ),
    ),
)
