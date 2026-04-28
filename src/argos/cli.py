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

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.command == "run":
        return asyncio.run(_run(args.url))
    return 1


if __name__ == "__main__":
    sys.exit(main())
