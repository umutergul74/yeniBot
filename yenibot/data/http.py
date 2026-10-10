"""Bounded retries and verified local caching of public Binance archive bytes."""
import hashlib
from pathlib import Path
from urllib.parse import urlparse
import uuid

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class ArchiveSession(requests.Session):
    """Single writer cache. Only successful ZIP responses are cached, never 404s."""

    def __init__(self, directory):
        super().__init__()
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        retry = Retry(total=3, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504),
                      allowed_methods=frozenset({"GET", "POST"}), respect_retry_after_header=False)
        self.mount("https://", HTTPAdapter(max_retries=retry))

    def get(self, url, **kwargs):
        parsed = urlparse(url)
        cacheable = (parsed.scheme == "https" and parsed.hostname in {
            "data.binance.vision", "s3-ap-northeast-1.amazonaws.com"
        } and parsed.path.endswith(".zip") and not parsed.query and not kwargs.get("params"))
        if not cacheable:
            return super().get(url, **kwargs)
        key = hashlib.sha256(url.encode()).hexdigest()
        path = self.directory / (key + ".zip")
        checksum = self.directory / (key + ".sha256")
        if path.exists() and checksum.exists():
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != checksum.read_text().strip():
                raise ValueError(f"Archive cache checksum mismatch: {path.name}")
            response = requests.Response()
            response.status_code, response.url, response._content = 200, url, content
            return response
        response = super().get(url, **kwargs)
        if response.status_code == 200:
            # Cache integrity only; origin is HTTPS, not authenticated by this hash.
            temporary = path.with_suffix("." + uuid.uuid4().hex + ".tmp")
            temporary.write_bytes(response.content)
            temporary.replace(path)
            checksum.write_text(hashlib.sha256(response.content).hexdigest(), encoding="ascii")
        return response
