from __future__ import annotations

import asyncio
import ipaddress
import socket
from urllib.parse import urlsplit

from lxml import etree
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright
from readability import Document

BLOCKED_RESOURCE_TYPES = {"image", "stylesheet", "font", "media"}
_ALLOWED_SCHEMES = {"http", "https"}
_BLOCKED_HOSTNAMES = {"localhost", "localhost.localdomain", "ip6-localhost"}
_BLOCKED_SUFFIXES = (".localhost", ".local", ".internal")


def _is_unsafe_ip(ip: ipaddress._BaseAddress) -> bool:
    return (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _resolve_hostname(host: str) -> list[ipaddress._BaseAddress]:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
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


def _is_safe_url(url: str) -> bool:
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
    resolved = _resolve_hostname(host_lower)
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
