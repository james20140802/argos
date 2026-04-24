from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket
from urllib.parse import urlsplit

from lxml import etree
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright
from readability import Document

from argos.crawler._robots import _robots_cache, is_robots_allowed
from argos.crawler.user_agents import random_user_agent

logger = logging.getLogger(__name__)

BLOCKED_RESOURCE_TYPES = {"image", "stylesheet", "font", "media"}
_ALLOWED_SCHEMES = {"http", "https"}
_BLOCKED_HOSTNAMES = {"localhost", "localhost.localdomain", "ip6-localhost"}
_BLOCKED_SUFFIXES = (".localhost", ".local", ".internal")

_is_robots_allowed = is_robots_allowed

__all__ = ["_robots_cache"]


def _is_unsafe_ip(ip: ipaddress._BaseAddress) -> bool:
    return not ip.is_global


async def _resolve_hostname(host: str) -> list[ipaddress._BaseAddress]:
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, None)
    except (socket.gaierror, UnicodeError):
        return []
    addresses: list[ipaddress._BaseAddress] = []
    for info in infos:
        sockaddr = info[4]
        if not sockaddr:
            continue
        try:
            addresses.append(ipaddress.ip_address(sockaddr[0]))
        except ValueError:
            continue
    return addresses


async def _is_safe_url(url: str) -> bool:
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    if parts.scheme not in _ALLOWED_SCHEMES:
        return False
    host = parts.hostname
    if not host:
        return False
    host_lower = host.lower().rstrip(".")
    if host_lower in _BLOCKED_HOSTNAMES:
        return False
    if any(host_lower.endswith(suffix) for suffix in _BLOCKED_SUFFIXES):
        return False
    try:
        literal_ip = ipaddress.ip_address(host_lower)
    except ValueError:
        literal_ip = None
    if literal_ip is not None:
        return not _is_unsafe_ip(literal_ip)
    resolved = await _resolve_hostname(host_lower)
    if not resolved:
        return False
    return not any(_is_unsafe_ip(ip) for ip in resolved)


def extract_main_content(html: str) -> tuple[str, str]:
    title = ""
    summary_html = html

    try:
        doc = Document(html)
        title = doc.title() or ""
        extracted = doc.summary()
        if extracted and extracted.strip():
            summary_html = extracted
    except Exception:
        summary_html = html

    for candidate in (summary_html, html):
        if not candidate or not candidate.strip():
            continue
        try:
            root = etree.fromstring(candidate.encode(), parser=etree.HTMLParser())
        except (etree.XMLSyntaxError, ValueError, TypeError):
            continue
        if root is None:
            continue
        return title, " ".join(root.itertext()).strip()

    return title, ""


async def fetch_dynamic_page(
    url: str,
    *,
    max_retries: int = 3,
    timeout_ms: int = 15000,
) -> dict | None:
    if not await _is_safe_url(url):
        return None
    if not await _is_robots_allowed(url):
        return None

    attempt = 0
    while attempt <= max_retries:
        try:
            html, final_url = await _load_page_html(url, timeout_ms)
            if final_url != url:
                if not await _is_safe_url(final_url):
                    logger.warning("SSRF redirect blocked: %s -> %s (failed _is_safe_url)", url, final_url)
                    return None
                if not await _is_robots_allowed(final_url):
                    logger.warning("SSRF redirect blocked: %s -> %s (failed _is_robots_allowed)", url, final_url)
                    return None
            title, raw_content = extract_main_content(html)
            return {"title": title, "source_url": url, "raw_content": raw_content}
        except PlaywrightTimeoutError as exc:
            attempt += 1
            if attempt > max_retries:
                logger.error("fetch_dynamic_page timed out after %d attempts for %s", max_retries + 1, url, exc_info=True)
                return None
            logger.warning("fetch_dynamic_page timeout (attempt %d/%d) for %s: %r", attempt, max_retries + 1, url, exc)
            await asyncio.sleep(2**attempt)
        except PlaywrightError as exc:
            attempt += 1
            if attempt > max_retries:
                logger.error("fetch_dynamic_page browser error after %d attempts for %s", max_retries + 1, url, exc_info=True)
                return None
            logger.warning("fetch_dynamic_page browser error (attempt %d/%d) for %s: %r", attempt, max_retries + 1, url, exc)
            await asyncio.sleep(2**attempt)


async def _load_page_html(url: str, timeout_ms: int) -> tuple[str, str]:
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        try:
            context = await browser.new_context(user_agent=random_user_agent())
            try:
                page = await context.new_page()
                try:
                    async def _route_handler(route):
                        request = route.request
                        if request.resource_type in BLOCKED_RESOURCE_TYPES:
                            await route.abort()
                            return
                        if not await _is_safe_url(request.url):
                            logger.warning(
                                "SSRF request blocked mid-flight: %s (resource_type=%s)",
                                request.url,
                                request.resource_type,
                            )
                            await route.abort()
                            return
                        await route.continue_()

                    await page.route("**/*", _route_handler)
                    await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                    final_url = page.url
                    html = await page.content()
                    return html, final_url
                finally:
                    await page.close()
            finally:
                await context.close()
        finally:
            await browser.close()
