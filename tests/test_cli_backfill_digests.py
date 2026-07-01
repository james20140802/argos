from __future__ import annotations

import argparse

from argos.cli import _build_backfill_digests_parser


def _parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command")
    common = argparse.ArgumentParser(add_help=False)
    _build_backfill_digests_parser(sub, common)
    return p


def test_backfill_digests_parses_defaults():
    args = _parser().parse_args(["backfill-digests"])
    assert args.command == "backfill-digests"
    assert args.limit is None
    assert args.dry_run is False


def test_backfill_digests_parses_flags():
    args = _parser().parse_args(["backfill-digests", "--limit", "5", "--dry-run"])
    assert args.limit == 5
    assert args.dry_run is True
