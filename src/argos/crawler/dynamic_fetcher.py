from __future__ import annotations

import asyncio
from urllib.parse import urlsplit

from lxml import etree
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright
from readability import Document

BLOCKED_RESOURCE_TYPES = {"image", "stylesheet", "font", "media"}
_ALLOWED_SCHEMES = {"http", "https"}


def _is_safe_url(url: str) -> bool:
    parts = urlsplit(url)
    if parts.scheme not in _ALLOWED_SCHEMES:
        return False
    if not parts.hostname:
        return False
    return True


def extract_main_content(html: str) -> tuple[str, str]:
    doc = Document(html)
    title = doc.title()
    summary_html = doc.summary()
    root = etree.fromstring(summary_html.encode(), parser=etree.HTMLParser())
    text = " ".join(root.itertext())
    return title, text.strip()


async def fetch_dynamic_page(
    url: str,
    *,
    max_retries: int = 3,
    timeout_ms: int = 15000,
) -> dict | None:
    if not _is_safe_url(url):
        return None

    attempt = 0
    while attempt <= max_retries:
        try:
            html = await _load_page_html(url, timeout_ms)
            title, raw_content = extract_main_content(html)
            return {"title": title, "source_url": url, "raw_content": raw_content}
        except (PlaywrightTimeoutError, PlaywrightError):
            attempt += 1
            if attempt > max_retries:
                return None
            await asyncio.sleep(2**attempt)


async def _load_page_html(url: str, timeout_ms: int) -> str:
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        try:
            context = await browser.new_context()
            try:
                page = await context.new_page()
                try:
                    async def _block_resources(route):
                        if route.request.resource_type in BLOCKED_RESOURCE_TYPES:
                            await route.abort()
                        else:
                            await route.continue_()

                    await page.route("**/*", _block_resources)
                    await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                    return await page.content()
                finally:
                    await page.close()
            finally:
                await context.close()
        finally:
            await browser.close()
