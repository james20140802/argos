from __future__ import annotations

import asyncio

from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

from argos.config import settings
from argos.slack.app import build_app


async def main() -> None:
    app = build_app()
    handler = AsyncSocketModeHandler(app, settings.secrets.SLACK_APP_TOKEN)
    await handler.start_async()


if __name__ == "__main__":
    asyncio.run(main())
