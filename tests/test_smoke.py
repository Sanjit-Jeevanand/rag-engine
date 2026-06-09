from rag_engine import config, log


def test_settings_defaults() -> None:
    assert config.settings.app_name == "rag-engine"
    assert config.settings.log_level == "INFO"


def test_bind_request_id_generates_uuid() -> None:
    rid = log.bind_request_id()
    assert len(rid) == 36
    assert rid.count("-") == 4
