from __future__ import annotations

from slack_bolt.async_app import AsyncApp

from argos.config import settings
from argos.slack.handlers.keep import handle_keep
from argos.slack.handlers.pass_ import handle_pass


def build_app() -> AsyncApp:
    if not settings.SLACK_BOT_TOKEN:
        raise ValueError("SLACK_BOT_TOKEN is not set")
    app = AsyncApp(token=settings.SLACK_BOT_TOKEN)
    register_handlers(app)
    return app


def register_handlers(app: AsyncApp) -> None:
    app.action("keep_tech")(handle_keep)
    app.action("pass_tech")(handle_pass)
