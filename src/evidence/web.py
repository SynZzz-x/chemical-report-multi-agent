from __future__ import annotations

import ipaddress
import socket
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

from .models import EvidenceRecord


def _host_matches(host: str, domain: str) -> bool:
    host = host.rstrip(".").lower()
    domain = domain.rstrip(".").lower()
    return host == domain or host.endswith(f".{domain}")


def _source_quality(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    if any(_host_matches(host, value) for value in ("gov.cn", "gov", "go.jp")):
        return "government"
    if any(_host_matches(host, value) for value in ("iso.org", "iec.ch", "standards.org")):
        return "standards"
    if any(
        _host_matches(host, value)
        for value in ("edu", "edu.cn", "doi.org", "arxiv.org")
    ):
        return "academic"
    return "vendor"


def _is_public_ip(value: str) -> bool:
    try:
        return ipaddress.ip_address(value).is_global
    except ValueError:
        return False


def _validate_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("web evidence URL must use HTTP(S)")
    if parsed.username or parsed.password:
        raise ValueError("web evidence URL must not contain credentials")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    addresses = socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)
    if not addresses or any(not _is_public_ip(item[4][0]) for item in addresses):
        raise ValueError("web evidence URL resolves to a non-public address")


def _safe_fetch_text(url: str, *, max_bytes: int = 2_000_000) -> tuple[str, str, str]:
    """Fetch a bounded public text page while validating every redirect."""
    import requests
    from bs4 import BeautifulSoup

    current = url
    for _ in range(4):
        _validate_public_url(current)
        response = requests.get(
            current,
            headers={
                "User-Agent": "Mozilla/5.0 evidence-fetcher/1.0",
                "Accept": "text/html,text/plain;q=0.9",
            },
            timeout=15,
            verify=True,
            allow_redirects=False,
            stream=True,
        )
        if response.status_code in {301, 302, 303, 307, 308}:
            location = response.headers.get("Location")
            response.close()
            if not location:
                raise ValueError("redirect response has no Location header")
            current = urljoin(current, location)
            continue
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower()
        if not any(value in content_type for value in ("text/html", "text/plain")):
            response.close()
            raise ValueError(f"unsupported web evidence content type: {content_type}")
        chunks: list[bytes] = []
        size = 0
        for chunk in response.iter_content(chunk_size=65536):
            if not chunk:
                continue
            size += len(chunk)
            if size > max_bytes:
                response.close()
                raise ValueError("web evidence page exceeds size limit")
            chunks.append(chunk)
        encoding = response.encoding or "utf-8"
        response.close()
        html = b"".join(chunks).decode(encoding, errors="replace")
        if "text/plain" in content_type:
            text = html
            title = current
        else:
            soup = BeautifulSoup(html, "html.parser")
            for element in soup(["script", "style", "noscript", "svg", "nav", "footer"]):
                element.decompose()
            title = soup.title.get_text(" ", strip=True) if soup.title else current
            text = soup.get_text("\n", strip=True)
        cleaned = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        if not cleaned:
            raise ValueError("web evidence page contains no usable text")
        return title, cleaned[:4000], current
    raise ValueError("web evidence URL exceeded redirect limit")


class LegacySpiderWebEvidenceProvider:
    """Use the existing search adapter, then safely fetch attributable pages."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        results_per_query: int = 3,
        allowed_source_classes: tuple[str, ...] = (
            "government", "standards", "academic", "vendor"
        ),
    ):
        self.output_dir = str(output_dir)
        self.results_per_query = max(1, results_per_query)
        self.allowed_source_classes = set(allowed_source_classes)

    def search(self, queries):
        from src.nodes.worker.tools.spider_final import WorkerScraper

        scraper = WorkerScraper(output_dir=self.output_dir)
        records: list[EvidenceRecord] = []
        accessed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        for query in queries:
            result = scraper.search_chemical_content(query, num_results=self.results_per_query)
            for item in result.get("search_results", []):
                url = str(item.get("url") or "").strip()
                quality = _source_quality(url)
                if quality not in self.allowed_source_classes:
                    continue
                try:
                    title, text, final_url = _safe_fetch_text(url)
                except Exception:
                    # Search snippets are discovery metadata, not claim evidence.
                    continue
                final_quality = _source_quality(final_url)
                if final_quality not in self.allowed_source_classes:
                    continue
                records.append(
                    EvidenceRecord(
                        evidence_id="pending",
                        source_type="web",
                        title=title or str(item.get("title") or final_url),
                        supporting_text=text,
                        url=final_url,
                        accessed_at=accessed_at,
                        source_quality=final_quality,
                        retrieval_query=str(item.get("search_query") or query),
                    )
                )
                # Ensure every uncovered-concept query gets a chance to run.
                break
        return tuple(records)
