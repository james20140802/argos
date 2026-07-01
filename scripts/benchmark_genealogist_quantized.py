"""Benchmark genealogist output quality for qwen3:32b-q4_K_M vs qwen3:32b.

Usage
-----
Run once per model and compare the JSON output files:

    uv run python scripts/benchmark_genealogist_quantized.py \\
        --model qwen3:32b --num-ctx 3072 --items 15 \\
        --out reports/genealogist-full.json

    uv run python scripts/benchmark_genealogist_quantized.py \\
        --model qwen3:32b-q4_K_M --num-ctx 6144 --items 15 \\
        --out reports/genealogist-q4km.json

See docs/benchmarks/genealogist-quantized.md for the full methodology,
scoring rubric, and VRAM/num_ctx probing procedure.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import select

# ---------------------------------------------------------------------------
# Reuse the canonical genealogist prompt — do not duplicate
# ---------------------------------------------------------------------------
from argos.brain._language import language_directive
from argos.brain.nodes.genealogist import _GENEALOGIST_PROMPT as GENEALOGIST_PROMPT
from argos.brain.ollama_client import LARGE_MODEL_TIMEOUT, query_ollama
from argos.database import AsyncSessionLocal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass (also the validated result of CLI arg-parsing)
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkArgs:
    """All knobs for a single benchmark run.

    Defaults are chosen so that ``BenchmarkArgs()`` describes the *baseline*
    run (full-precision model, current genealogist num_ctx).
    """

    model: str = "qwen3:32b"
    num_ctx: int = 3072
    items: int = 15
    out: Path = field(default_factory=lambda: Path("reports/genealogist-benchmark.json"))


# ---------------------------------------------------------------------------
# Core benchmark logic
# ---------------------------------------------------------------------------

async def run_benchmark(
    *,
    session: Any,
    model: str,
    num_ctx: int,
    items: int,
) -> dict[str, Any]:
    """Run the genealogist prompt against `items` tech_items from the DB.

    Returns a report dict with shape::

        {
          "model": str,
          "num_ctx": int,
          "items_evaluated": int,
          "results": [
            {
              "item_id": str,
              "title": str,
              "relation_type": str | None,
              "reason": str,
              "elapsed_s": float,
              # Only present on error:
              "error": str,
            },
            ...
          ],
        }

    The function never raises — per-item errors are recorded in the ``error``
    key of the corresponding result record so the full run completes even if
    one item triggers a GPU OOM or a parse failure.
    """
    from argos.models.tech_item import TechItem  # local import to avoid DB init at import time

    # Fetch `items` tech_items ordered by created_at DESC so we always bench
    # on recent, real content rather than stale seed data.
    stmt = (
        select(TechItem.id, TechItem.title, TechItem.raw_content)
        .order_by(TechItem.created_at.desc())
        .limit(items)
    )
    result = await session.execute(stmt)
    rows = result.fetchall()

    results: list[dict[str, Any]] = []

    for row in rows:
        item_id = str(row.id)
        title = row.title or "Untitled"
        raw_content = row.raw_content or ""

        # Build the prompt the same way genealogist_node does for a single-item run
        from argos.config import settings as _benchmark_settings

        _language = _benchmark_settings.user.slack.summary_language or "English"
        prompt = GENEALOGIST_PROMPT.format(
            new_tech=raw_content[:1000],
            existing_techs="(benchmark mode — no similarity context)",
            language=_language,
            language_reminder=language_directive(_language),
        )

        start = time.perf_counter()
        try:
            raw = await query_ollama(
                model,
                prompt,
                keep_alive=0,  # unload between items so VRAM resets cleanly
                timeout=LARGE_MODEL_TIMEOUT,
                num_ctx=num_ctx,
                think=False,
            )
            elapsed = time.perf_counter() - start

            # Parse JSON from response
            json_start = raw.find("{")
            json_end = raw.rfind("}") + 1
            if json_start == -1 or json_end == 0:
                raise ValueError(f"No JSON object found in response: {raw[:200]!r}")
            parsed = json.loads(raw[json_start:json_end])

            results.append(
                {
                    "item_id": item_id,
                    "title": title,
                    "relation_type": parsed.get("relation_type"),
                    "reason": parsed.get("reason", ""),
                    "elapsed_s": round(elapsed, 3),
                }
            )
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - start
            logger.warning("Item %s (%s) failed: %r", item_id, title, exc)
            results.append(
                {
                    "item_id": item_id,
                    "title": title,
                    "relation_type": None,
                    "reason": "",
                    "elapsed_s": round(elapsed, 3),
                    "error": str(exc),
                }
            )

    return {
        "model": model,
        "num_ctx": num_ctx,
        "items_evaluated": len(results),
        "results": results,
    }


def write_report(report: dict[str, Any], path: Path) -> None:
    """Write `report` as pretty-printed JSON to `path`, creating parents as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    logger.info("Report written to %s", path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> BenchmarkArgs:
    parser = argparse.ArgumentParser(
        description="Benchmark genealogist output quality across Ollama models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        default="qwen3:32b",
        help="Ollama model tag to benchmark (default: qwen3:32b)",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=3072,
        dest="num_ctx",
        help="Context window size (default: 3072 — matches current genealogist default)",
    )
    parser.add_argument(
        "--items",
        type=int,
        default=15,
        help="Number of tech_items to evaluate (default: 15)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/genealogist-benchmark.json"),
        help="Output path for the JSON report (default: reports/genealogist-benchmark.json)",
    )
    ns = parser.parse_args()
    return BenchmarkArgs(model=ns.model, num_ctx=ns.num_ctx, items=ns.items, out=ns.out)


async def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()

    logger.info(
        "Starting benchmark: model=%s num_ctx=%d items=%d out=%s",
        args.model,
        args.num_ctx,
        args.items,
        args.out,
    )

    async with AsyncSessionLocal() as session:
        report = await run_benchmark(
            session=session,
            model=args.model,
            num_ctx=args.num_ctx,
            items=args.items,
        )

    write_report(report, args.out)
    print(f"Done. {report['items_evaluated']} items evaluated.")
    print(f"Report: {args.out}")


if __name__ == "__main__":
    asyncio.run(_main())
