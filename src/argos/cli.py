from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time

from argos.crawler.pipeline import run_full_pipeline
from argos.database import AsyncSessionLocal


def _format_duration(seconds: float) -> str:
    total = int(seconds)
    minutes, secs = divmod(total, 60)
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


async def _run(dynamic_urls: list[str] | None) -> int:
    start = time.monotonic()
    async with AsyncSessionLocal() as session:
        results, summary = await run_full_pipeline(session, dynamic_urls=dynamic_urls or None)
    elapsed = time.monotonic() - start

    # Build per-source breakdown string
    source_parts = []
    if "github_trending" in summary.per_source:
        source_parts.append(f"GitHub: {summary.per_source['github_trending']}")
    if "hackernews" in summary.per_source:
        source_parts.append(f"HN: {summary.per_source['hackernews']}")
    for src, cnt in summary.per_source.items():
        if src not in ("github_trending", "hackernews"):
            source_parts.append(f"{src}: {cnt}")
    source_detail = f" ({', '.join(source_parts)})" if source_parts else ""

    print("✅ argos run 완료")
    print("─────────────────────────────")
    print(f"크롤링: {summary.crawled_total}개{source_detail}")
    print(f"트리아지 통과: {summary.triage_pass}개")
    print(f"신규 저장: {summary.saved_new}개")
    print(f"소요 시간: {_format_duration(elapsed)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="argos")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run the full crawl → brain → save pipeline")
    run_p.add_argument(
        "--url",
        action="append",
        default=[],
        help="Extra dynamic URL to fetch (repeatable)",
    )
    run_p.add_argument("-v", "--verbose", action="store_true")

    sub.add_parser("slack", help="Start the Slack bot (Socket Mode)")

    brief_p = sub.add_parser("brief", help="Dispatch today's briefing to Slack")
    brief_p.add_argument("--channel", default=None, help="Override target Slack channel ID")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.command == "run":
        return asyncio.run(_run(args.url))
    if args.command == "slack":
        from argos.main import main as slack_main

        asyncio.run(slack_main())
        return 0
    if args.command == "brief":
        from argos.slack.briefing import dispatch_daily_briefing

        ts = asyncio.run(dispatch_daily_briefing(channel=args.channel))
        if ts:
            print(f"Briefing sent: ts={ts}")
        else:
            print("No items today — briefing skipped")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
