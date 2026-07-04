"""Shared fixtures for ``tests/init_wizard`` (ARG-191).

Two regression tests here — ``test_rebuild_database_uses_new_port_from_env``
and ``test_run_reconfigure_slack_rebuilds_db_engine_from_env_path`` — call
``wizard._rebuild_database`` / ``argos.database.rebuild()`` with a throwaway
``.env`` pointing at a bogus port (9999) to verify the engine-rebuild wiring.

``rebuild()`` mutates *module-level* global state with no built-in teardown:

- ``argos.database.engine`` / ``argos.database.AsyncSessionLocal`` are
  replaced with a new engine bound to the bogus port.
- ``argos.config.settings.secrets`` is replaced with a freshly constructed
  ``Secrets()``.
- ``load_dotenv(override=True)`` permanently writes the bogus ``POSTGRES_*``
  values into ``os.environ`` for the rest of the process.

Left unrestored, this leaks into every test that runs afterward in the same
pytest session: any DB-backed test that goes through the *global*
``AsyncSessionLocal`` (rather than creating its own engine from a captured
DB URL) silently starts talking to the bogus port and fails. This was the
root cause of the order-dependent flakiness in
``tests/test_cli_backfill_images_db.py::test_backfill_upgrade_favicons``
(only reproduced when the full suite ran, because ``init_wizard/`` collects
— and therefore executes — before that file alphabetically).

This autouse fixture snapshots and restores all three pieces of global state
around every test in this package, so `rebuild()` exercises are fully
self-contained regardless of run order.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _restore_database_globals():
    import argos.database as db_module
    from argos.config import settings

    original_engine = db_module.engine
    original_session_local = db_module.AsyncSessionLocal
    original_secrets = settings.secrets
    original_environ = dict(os.environ)

    yield

    db_module.engine = original_engine
    db_module.AsyncSessionLocal = original_session_local
    settings.secrets = original_secrets
    os.environ.clear()
    os.environ.update(original_environ)
