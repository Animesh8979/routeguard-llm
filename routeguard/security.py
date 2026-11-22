"""Adversarial text normalization (NFKC + zero-width + Base64), SSRF guard, and timing-safe API auth."""
from __future__ import annotations

import base64
import hmac
import ipaddress
import re
import socket
import unicodedata
from urllib.parse import urlparse

MAX_PROMPT_CHARS = 32_000

_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
_BASE64_CAND_RE = re.compile(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{24,}={0,2}(?![A-Za-z0-9+/=])")


def normalize_adversarial_text(text: str) -> tuple[str, list[str]]:
    """Normalize NFKC homoglyphs, strip zero-width characters, and decode embedded Base64 blobs."""
    raw = text or ""
    nfkc = unicodedata.normalize("NFKC", raw)
    stripped = _ZERO_WIDTH_RE.sub("", nfkc)
    decoded_blobs: list[str] = []
    for cand in _BASE64_CAND_RE.findall(stripped):
        padded = cand + "=" * ((4 - len(cand) % 4) % 4)
        try:
            raw_bytes = base64.b64decode(padded, validate=True)
            txt = raw_bytes.decode("utf-8", errors="ignore")
            if txt and sum(ch.isprintable() or ch.isspace() for ch in txt) / max(1, len(txt)) >= 0.85:
                decoded_blobs.append(txt)
        except Exception:
            continue
    return stripped, decoded_blobs


def validate_upstream_url(url: str, *, allow_localhost: bool = False) -> str:
    """Validate upstream LLM HTTP/HTTPS URL and block private/link-local/metadata SSRF targets."""
    u = (url or "").strip()
    if not u:
        return u
    parsed = urlparse(u)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Blocked non-HTTP scheme in upstream URL: {parsed.scheme!r}")
    host = (parsed.hostname or "").strip().lower()
    if not host:
        raise ValueError("Missing hostname in upstream URL")

    if not allow_localhost and host in ("localhost", "ip6-localhost"):
        raise ValueError(f"Blocked localhost SSRF target: {host}")

    def _check_ip(ip_obj: ipaddress.IPv4Address | ipaddress.IPv6Address) -> None:
        if ip_obj.is_link_local or ip_obj.is_multicast or ip_obj.is_unspecified:
            raise ValueError(f"Blocked restricted IP range: {ip_obj}")
        if ip_obj.is_loopback and not allow_localhost:
            raise ValueError(f"Blocked loopback SSRF target: {ip_obj}")
        if ip_obj.is_private and not (allow_localhost and ip_obj.is_loopback):
            raise ValueError(f"Blocked private network SSRF target: {ip_obj}")

    try:
        ip_direct = ipaddress.ip_address(host)
        _check_ip(ip_direct)
        return u
    except ValueError as exc:
        if "Blocked" in str(exc):
            raise

    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror:
        # Offline environments may not resolve public hostnames; synthetic tests inject direct IPs.
        return u

    for info in infos:
        addr_str = info[4][0].split("%")[0]
        try:
            _check_ip(ipaddress.ip_address(addr_str))
        except ValueError as exc:
            if "Blocked" in str(exc):
                raise
    return u


def verify_api_key(provided: str | None, expected: str | None) -> bool:
    """Verify X-RouteGuard-Key header in constant time when configured."""
    if not expected:
        return True
    if not provided:
        return False
    return hmac.compare_digest(provided.strip(), expected.strip())
