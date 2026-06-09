from pathlib import Path

from rag_engine.ingest.parser import parse_snapshot

FIXTURE = Path(__file__).parent / "fixtures" / "sample_snapshot.jsonl"


def test_parses_valid_articles() -> None:
    articles = list(parse_snapshot(FIXTURE))
    assert len(articles) == 3


def test_article_fields_non_empty() -> None:
    articles = list(parse_snapshot(FIXTURE))
    for article in articles:
        assert article.article_id
        assert article.title
        assert article.text
        assert article.timestamp


def test_first_article_values() -> None:
    articles = list(parse_snapshot(FIXTURE))
    assert articles[0].article_id == "12"
    assert articles[0].title == "Anarchism"


def test_skips_non_main_namespace() -> None:
    articles = list(parse_snapshot(FIXTURE))
    ids = [a.article_id for a in articles]
    assert "0" not in ids  # namespace=1 talk page excluded
    assert "12" in ids
    assert "39" in ids
    assert len(articles) == 3


def test_categories_joined_as_string() -> None:
    articles = list(parse_snapshot(FIXTURE))
    assert articles[0].categories == "Political ideologies Anarchism"
