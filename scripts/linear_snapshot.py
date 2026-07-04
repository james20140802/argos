#!/usr/bin/env python3
"""Linear 상태 스냅샷 — 루틴/에이전트가 MCP 왕복 없이 한 번에 큐를 읽기 위한 도구.

MCP list_issues/get_issue 응답은 안 쓰는 필드까지 통째로 컨텍스트에 들어가므로,
나이틀리 루틴(A: Todo 스캔, B: Ready 큐)은 이 스크립트로 필요한 필드만 담은
압축 JSON을 Bash 한 번으로 얻는다.

사용 예:
    export ARGOS_LINEAR_API_KEY=...   # ~/.zshrc에 있음
    python3 scripts/linear_snapshot.py --state Ready                  # Routine B 큐
    python3 scripts/linear_snapshot.py --state Todo --with-description  # Routine A 대상
    python3 scripts/linear_snapshot.py --label auto-decomposed --state Backlog Ready
    python3 scripts/linear_snapshot.py --issue ARG-173                # 단일 이슈 + 코멘트

출력: 한 줄 JSON (stdout). 오류는 stderr + exit 1.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

API_URL = "https://api.linear.app/graphql"
DEFAULT_TEAM = "Argos"  # 팀 이름 (Sangchu는 워크스페이스 이름, 팀 key는 ARG)

LIST_QUERY = """
query($filter: IssueFilter, $after: String) {
  issues(filter: $filter, first: 100, after: $after) {
    pageInfo { hasNextPage endCursor }
    nodes {
      identifier title url description
      state { name }
      assignee { displayName }
      labels { nodes { name } }
      parent { identifier }
      children { nodes { identifier } }
      relations { nodes { type relatedIssue { identifier } } }
      inverseRelations { nodes { type issue { identifier } } }
    }
  }
}
"""

ISSUE_QUERY = """
query($id: String!) {
  issue(id: $id) {
    identifier title url description
    state { name }
    assignee { displayName }
    labels { nodes { name } }
    parent { identifier title }
    children { nodes { identifier title state { name } } }
    relations { nodes { type relatedIssue { identifier } } }
    inverseRelations { nodes { type issue { identifier } } }
    comments(first: 50) { nodes { createdAt user { displayName } body } }
  }
}
"""


def gql(query: str, variables: dict) -> dict:
    key = os.environ.get("ARGOS_LINEAR_API_KEY")
    if not key:
        sys.exit("ARGOS_LINEAR_API_KEY가 환경에 없습니다 (~/.zshrc 참고)")
    req = urllib.request.Request(
        API_URL,
        data=json.dumps({"query": query, "variables": variables}).encode(),
        headers={"Content-Type": "application/json", "Authorization": key},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.load(resp)
    if body.get("errors"):
        sys.exit(f"Linear API 오류: {body['errors']}")
    return body["data"]


def compact_issue(n: dict, with_description: bool) -> dict:
    out = {
        "id": n["identifier"],
        "title": n["title"],
        "state": n["state"]["name"],
        "labels": [lb["name"] for lb in n["labels"]["nodes"]],
        "assignee": (n.get("assignee") or {}).get("displayName"),
        "parent": (n.get("parent") or {}).get("identifier"),
        "children": [c["identifier"] for c in n.get("children", {}).get("nodes", [])],
        "blocks": [
            r["relatedIssue"]["identifier"]
            for r in n.get("relations", {}).get("nodes", [])
            if r["type"] == "blocks" and r.get("relatedIssue")
        ],
        "blocked_by": [
            r["issue"]["identifier"]
            for r in n.get("inverseRelations", {}).get("nodes", [])
            if r["type"] == "blocks" and r.get("issue")
        ],
        "url": n["url"],
    }
    if with_description:
        out["description"] = n.get("description")
    return {k: v for k, v in out.items() if v not in (None, [], "")}


def list_issues(args: argparse.Namespace) -> dict:
    flt: dict = {"team": {"name": {"eq": args.team}}}
    if args.state:
        flt["state"] = {"name": {"in": args.state}}
    if args.label:
        flt["labels"] = {"some": {"name": {"in": args.label}}}
    nodes, after = [], None
    while True:
        data = gql(LIST_QUERY, {"filter": flt, "after": after})["issues"]
        nodes.extend(data["nodes"])
        if not data["pageInfo"]["hasNextPage"]:
            break
        after = data["pageInfo"]["endCursor"]
    return {
        "team": args.team,
        "filter": {"state": args.state, "label": args.label},
        "count": len(nodes),
        "issues": [compact_issue(n, args.with_description) for n in nodes],
    }


def single_issue(args: argparse.Namespace) -> dict:
    n = gql(ISSUE_QUERY, {"id": args.issue})["issue"]
    if n is None:
        sys.exit(f"이슈를 찾을 수 없음: {args.issue}")
    out = compact_issue(n, with_description=True)
    out["children"] = [
        {"id": c["identifier"], "title": c["title"], "state": c["state"]["name"]}
        for c in n["children"]["nodes"]
    ]
    out["comments"] = [
        {
            "at": c["createdAt"][:16],
            "by": (c.get("user") or {}).get("displayName"),
            "body": c["body"],
        }
        for c in n["comments"]["nodes"]
    ]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--team", default=DEFAULT_TEAM)
    p.add_argument("--state", nargs="*", help="상태 이름 필터 (예: Ready Todo)")
    p.add_argument("--label", nargs="*", help="라벨 이름 필터 (예: auto-decomposed)")
    p.add_argument("--with-description", action="store_true", help="본문 포함 (기본 제외)")
    p.add_argument("--issue", help="단일 이슈 identifier (예: ARG-173) — 본문+코멘트 포함")
    args = p.parse_args()

    result = single_issue(args) if args.issue else list_issues(args)
    json.dump(result, sys.stdout, ensure_ascii=False)
    print()


if __name__ == "__main__":
    main()
