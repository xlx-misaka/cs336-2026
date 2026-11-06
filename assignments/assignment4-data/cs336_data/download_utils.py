"""Download infrastructure for large gzip assets."""

import gzip
import http.client
import logging
import shutil
import ssl
import tempfile
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path


def _validate_gzip(path: Path) -> None:
    """Read to EOF to verify gzip framing, length, and checksums."""
    with gzip.open(path, "rb") as stream:
        while stream.read(1024 * 1024):
            pass


def download_gzip_with_retries(
    url: str,
    destination: Path,
    *,
    logger: logging.Logger | None = None,
    attempts: int = 8,
    timeout: float = 60,
) -> Path:
    """Reuse valid caches and publish downloads only after integrity checks pass."""
    if attempts < 1:
        raise ValueError("attempts must be positive")
    logger = logger or logging.getLogger(__name__)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        try:
            _validate_gzip(destination)
        except (EOFError, gzip.BadGzipFile, zlib.error) as error:
            invalid = destination.with_name(f"{destination.name}.invalid-{time.time_ns()}")
            destination.replace(invalid)
            logger.warning("Invalid gzip cache moved to %s: %s", invalid, error)
        else:
            logger.info("Using validated WET cache %s", destination)
            return destination

    for attempt in range(1, attempts + 1):
        partial = None
        try:
            logger.info("Downloading %s to %s (attempt %d/%d)", url, destination, attempt, attempts)
            with tempfile.NamedTemporaryFile(
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".part",
                delete=False,
            ) as handle:
                partial = Path(handle.name)
            with urllib.request.urlopen(url, timeout=timeout) as response, partial.open("wb") as output:
                expected = response.headers.get("Content-Length")
                shutil.copyfileobj(response, output, length=1024 * 1024)
            actual = partial.stat().st_size
            if expected is not None and actual != int(expected):
                raise urllib.error.ContentTooShortError(
                    f"retrieval incomplete: got {actual} out of {expected} bytes", None
                )
            _validate_gzip(partial)
            partial.replace(destination)
            logger.info("Downloaded and validated %s (%d bytes)", destination, actual)
            return destination
        except urllib.error.HTTPError as error:
            error.close()
            if error.code not in (408, 429, 500, 502, 503, 504):
                raise
            failure = error
        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            ssl.SSLError,
            http.client.IncompleteRead,
            EOFError,
            gzip.BadGzipFile,
            zlib.error,
        ) as error:
            failure = error
        finally:
            if partial is not None:
                partial.unlink(missing_ok=True)

        if attempt == attempts:
            raise failure
        delay = min(2**attempt, 30)
        logger.warning("Download attempt %d/%d failed: %s; retrying in %ds", attempt, attempts, failure, delay)
        time.sleep(delay)

    raise AssertionError("unreachable")
