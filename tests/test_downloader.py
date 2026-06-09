import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from rag_engine.ingest.downloader import download_snapshot


def test_skips_if_exists() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "snapshot.json"
        dest.write_text("existing")
        result = download_snapshot(dest)
        assert result == dest
        assert dest.read_text() == "existing"  # untouched


def test_writes_to_tmp_then_renames(tmp_path: Path) -> None:
    dest = tmp_path / "snapshot.json"

    mock_response = MagicMock()
    mock_response.iter_bytes.return_value = [b"chunk1", b"chunk2"]
    mock_response.__enter__ = lambda s: s
    mock_response.__exit__ = MagicMock(return_value=False)

    mock_client = MagicMock()
    mock_client.stream.return_value = mock_response
    mock_client.__enter__ = lambda s: s
    mock_client.__exit__ = MagicMock(return_value=False)

    with patch("rag_engine.ingest.downloader.httpx.Client", return_value=mock_client):
        download_snapshot(dest)

    assert dest.exists()
    assert not dest.with_suffix(".tmp").exists()
    assert dest.read_bytes() == b"chunk1chunk2"
