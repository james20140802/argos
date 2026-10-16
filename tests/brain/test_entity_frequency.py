from argos.brain.entity_frequency import drop_common_names


def _docs(n_openai: int, n_kimi: int, total: int) -> dict[int, frozenset[str]]:
    docs: dict[int, frozenset[str]] = {}
    for index in range(total):
        names: set[str] = {f"unique {index}"}
        if index < n_openai:
            names.add("openai")
        if index < n_kimi:
            names.add("kimi k3")
        docs[index] = frozenset(names)
    return docs


def test_drop_common_names_removes_keys_over_ratio():
    docs = _docs(6, 1, 10)
    result = drop_common_names(
        docs,
        document_frequency={"openai": 6, "kimi k3": 1},
        named_documents=10,
        max_ratio=0.3,
    )
    assert all("openai" not in names for names in result.values())
    assert "kimi k3" in result[0]
    assert "unique 0" in result[0]


def test_zero_named_documents_returns_input():
    docs = {1: frozenset({"openai"})}
    result = drop_common_names(
        docs, document_frequency={}, named_documents=0, max_ratio=0.1
    )
    assert result == docs


def test_ratio_uses_named_documents_not_all():
    docs = _docs(6, 0, 10)
    kwargs = dict(document_frequency={"openai": 6}, max_ratio=0.3)
    # 이름 있는 문서가 10개면 6/10 > 0.3 이라 빠지지만, 분모가 훨씬 크면(전체
    # 기사 수로 나누는 잘못된 계산) 남는다 — 분모는 호출자가 준 값 그대로다.
    dropped = drop_common_names(docs, named_documents=10, **kwargs)
    kept = drop_common_names(docs, named_documents=100, **kwargs)
    assert "openai" not in dropped[0]
    assert "openai" in kept[0]


def test_ratio_boundary_is_exclusive():
    docs = _docs(3, 0, 10)
    result = drop_common_names(
        docs, document_frequency={"openai": 3}, named_documents=10, max_ratio=0.3
    )
    assert "openai" in result[0]


def test_deterministic():
    docs = _docs(6, 2, 10)
    kwargs = dict(
        document_frequency={"openai": 6, "kimi k3": 2}, named_documents=10, max_ratio=0.3
    )
    assert drop_common_names(docs, **kwargs) == drop_common_names(docs, **kwargs)
