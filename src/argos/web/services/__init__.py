"""Service layer for the Argos web module.

Services are pure-async functions that take an ``AsyncSession`` plus
keyword args and return plain dataclasses. They never touch
``tech_items.briefed_at`` — that column is exclusively owned by the
Slack briefing pipeline (ARG-136 decision).
"""
