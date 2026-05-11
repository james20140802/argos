from __future__ import annotations

from slack_bolt.async_app import AsyncApp

from argos.config import settings
from argos.slack.handlers.deep_dive import handle_deep_dive
from argos.slack.handlers.keep import handle_keep
from argos.slack.handlers.pass_ import handle_pass
from argos.slack.handlers.portfolio import handle_portfolio_command, handle_portfolio_mention
from argos.slack.handlers.untrack import handle_untrack


def build_app() -> AsyncApp:
    if not settings.secrets.SLACK_BOT_TOKEN:
        raise ValueError("SLACK_BOT_TOKEN is not set")
    app = AsyncApp(token=settings.secrets.SLACK_BOT_TOKEN)
    register_handlers(app)
    return app


def register_handlers(app: AsyncApp) -> None:
    app.action("action_keep")(handle_keep)
    app.action("action_pass")(handle_pass)
    app.action("action_deep_dive")(handle_deep_dive)
    app.action("action_untrack")(handle_untrack)
    app.command("/argos")(handle_portfolio_command)
    app.event("app_mention")(handle_portfolio_mention)
