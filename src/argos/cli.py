from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from argos.crawler.pipeline import run_full_pipeline
from argos.database import AsyncSessionLocal


async def _run(dynamic_urls: list[str] | None) -> int:
    async with AsyncSessionLocal() as session:
        results = await run_full_pipeline(session, dynamic_urls=dynamic_urls or None)
    print(f"처리 완료: {len(results)}개 항목")
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
