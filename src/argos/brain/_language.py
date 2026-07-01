"""Shared strong language directive for brain LLM prompts (ARG-173 / T5).

The weak mid-prompt "written in {language}" clause was being ignored by
qwen3 when the source text was English — the model mirrored the source
language. Appending an emphatic directive AFTER the source text uses recency
to override that bias. Reused by triage, digest, genealogist, and deep dive.
"""
from __future__ import annotations


def language_directive(language: str) -> str:
    """Return an emphatic trailing block forcing output language.

    Meant to be concatenated at the very END of a prompt (after the source
    text) so it is the last instruction the model reads.
    """
    return (
        f"\n\nIMPORTANT: Write every natural-language output field "
        f"(summary, reason, digest, analysis) in {language} ONLY, "
        f"regardless of the language of the source text above."
    )
