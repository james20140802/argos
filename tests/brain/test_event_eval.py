"""ARG-294: 판정 평가 지표 — DB 없이 돈다."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from argos.brain.event_eval import (
    PLATFORM_DOMAINS,
    JudgedCluster,
    JudgmentSet,
    evaluate,
    format_report,
    load_judgments,
    platform_pairs_together,
)

DATA = Path(__file__).resolve().parents[2] / "evals" / "event_judgments_arg283.json"


def _ids(count: int) -> tuple[uuid.UUID, ...]:
    return tuple(uuid.UUID(int=index + 1) for index in range(count))


def _set(*clusters: JudgedCluster, must_keep: tuple[int, ...] = ()) -> JudgmentSet:
    return JudgmentSet(clusters=clusters, must_keep=must_keep)


def test_committed_data_has_128_clusters_with_expected_labels():
    judgments = load_judgments(DATA)
    labels = [cluster.label for cluster in judgments.clusters]
    assert len(labels) == 128
    assert (labels.count("ok"), labels.count("maybe"), labels.count("bad")) == (45, 28, 55)
    assert judgments.must_keep == (21, 25)
    assert all(len(cluster.doc_ids) >= 2 for cluster in judgments.clusters)


def test_committed_data_carries_no_text_fields():
    raw = json.loads(DATA.read_text(encoding="utf-8"))
    for cluster in raw["clusters"]:
        assert set(cluster) == {"n", "label", "doc_ids"}
    assert "미검증" in raw["description"]


def test_unchanged_partition_reproduces_baseline_counts():
    judgments = load_judgments(DATA)
    # 판정 당시처럼 각 묶음이 그대로 한 그룹인 분할.
    partition = {
        doc_id: cluster.n for cluster in judgments.clusters for doc_id in cluster.doc_ids
    }
    report = evaluate(judgments, partition)
    assert (report.ok_merged, report.maybe_merged, report.bad_merged) == (45, 28, 55)
    assert report.ok_strict == 45
    assert report.wrong_ratio == pytest.approx(55 / 100)
    assert report.wrong_ratio_maybe_as_wrong == pytest.approx(83 / 128)
    assert dict(report.must_keep) == {21: True, 25: True}
    assert report.missing_docs == 0
    assert report.multi_doc_groups == 128


def test_bad_cluster_fully_split_is_no_longer_merged():
    a, b, c = _ids(3)
    judgments = _set(JudgedCluster(n=1, label="bad", doc_ids=(a, b, c)))
    report = evaluate(judgments, {a: "x", b: "y", c: "z"})
    assert report.bad_merged == 0
    assert report.wrong_ratio is None  # 분모 0


def test_bad_cluster_with_two_left_together_is_still_merged():
    a, b, c = _ids(3)
    judgments = _set(JudgedCluster(n=1, label="bad", doc_ids=(a, b, c)))
    report = evaluate(judgments, {a: "x", b: "x", c: "z"})
    assert report.bad_merged == 1


def test_ok_strict_needs_every_member_together_loose_needs_two():
    a, b, c = _ids(3)
    judgments = _set(JudgedCluster(n=7, label="ok", doc_ids=(a, b, c)), must_keep=(7,))
    report = evaluate(judgments, {a: "x", b: "x", c: "z"})
    assert report.ok_merged == 1
    assert report.ok_strict == 0
    assert dict(report.must_keep) == {7: False}


def test_missing_documents_count_as_alone_and_are_reported():
    a, b = _ids(2)
    judgments = _set(JudgedCluster(n=1, label="ok", doc_ids=(a, b)))
    report = evaluate(judgments, {a: "x"})
    assert report.missing_docs == 1
    assert report.ok_merged == 0


def test_two_missing_documents_are_not_grouped_together():
    a, b = _ids(2)
    judgments = _set(JudgedCluster(n=1, label="bad", doc_ids=(a, b)))
    report = evaluate(judgments, {})
    assert report.bad_merged == 0
    assert report.missing_docs == 2


def test_multi_doc_groups_counts_whole_partition():
    a, b, c, d, e = _ids(5)
    judgments = _set(JudgedCluster(n=1, label="ok", doc_ids=(a, b)))
    report = evaluate(judgments, {a: 1, b: 1, c: 2, d: 2, e: 3})
    assert report.multi_doc_groups == 2


def test_format_report_marks_unverified_and_shows_ratio():
    a, b = _ids(2)
    judgments = _set(JudgedCluster(n=1, label="bad", doc_ids=(a, b)))
    text = format_report(evaluate(judgments, {a: 1, b: 1}), title="기준선")
    assert "Claude 판정 기준(미검증)" in text
    assert "기준선" in text
    assert "1/1" in text


def test_evaluate_is_deterministic():
    judgments = load_judgments(DATA)
    partition = {
        doc_id: cluster.n for cluster in judgments.clusters for doc_id in cluster.doc_ids
    }
    assert evaluate(judgments, partition) == evaluate(judgments, dict(reversed(partition.items())))


def test_platform_pairs_counts_co_assigned_same_platform_pairs():
    a, b, c, d = _ids(4)
    partition = {a: 1, b: 1, c: 1, d: 2}
    sources = {a: "github.com", b: "github.com", c: "github.com", d: "github.com"}
    # 그룹 1 안 github 쌍: (a,b),(a,c),(b,c) = 3. d는 따로.
    assert platform_pairs_together(partition, sources) == 3


def test_platform_pairs_ignore_non_platform_and_mixed_domains():
    a, b, c = _ids(3)
    partition = {a: 1, b: 1, c: 1}
    sources = {a: "openai.com", b: "openai.com", c: "github.com"}
    assert platform_pairs_together(partition, sources) == 0
    assert PLATFORM_DOMAINS == frozenset({"arxiv.org", "github.com"})


def test_platform_pairs_do_not_mix_platforms_or_count_unknown_sources():
    a, b, c, d = _ids(4)
    partition = {a: 1, b: 1, c: 1, d: 1}
    sources = {a: "github.com", b: "arxiv.org", c: None}  # d는 sources에 없음
    assert platform_pairs_together(partition, sources) == 0


def _write_judgments(tmp_path: Path, label: str, doc_ids: list[str]) -> Path:
    path = tmp_path / "judgments.json"
    path.write_text(
        json.dumps({"clusters": [{"n": 1, "label": label, "doc_ids": doc_ids}]}),
        encoding="utf-8",
    )
    return path


def test_load_judgments_rejects_an_unknown_label(tmp_path):
    ids = [str(value) for value in _ids(2)]
    with pytest.raises(ValueError, match="ok/maybe/bad"):
        load_judgments(_write_judgments(tmp_path, "okay", ids))


def test_load_judgments_rejects_a_cluster_with_fewer_than_two_docs(tmp_path):
    ids = [str(value) for value in _ids(1)]
    with pytest.raises(ValueError, match="최소 2개"):
        load_judgments(_write_judgments(tmp_path, "ok", ids))
