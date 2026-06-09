import logging
from pathlib import Path

import httpx

DOWNLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MB

logger = logging.getLogger(__name__)

WIKIPEDIA_DUMP_URL = (
    "https://dumps.wikimedia.org/other/cirrussearch/20251229/"
    "enwiki-20251229-cirrussearch-content.json.gz"
)


def download_snapshot(dest: Path, url: str = WIKIPEDIA_DUMP_URL) -> Path:
    if dest.exists():
        logger.info("snapshot already exists, skipping", extra={"path": str(dest)})
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")

    logger.info("downloading snapshot", extra={"url": url, "dest": str(dest)})

    with (
        httpx.Client(follow_redirects=True, timeout=300) as client,
        client.stream("GET", url) as response,
    ):
        response.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in response.iter_bytes(chunk_size=DOWNLOAD_CHUNK_SIZE):
                f.write(chunk)

    tmp.rename(dest)
    logger.info("download complete", extra={"path": str(dest)})
    return dest
