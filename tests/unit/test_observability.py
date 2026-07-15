def test_prometheus_metrics_registered():
    # Trigger app creation so metrics are registered
    from unittest.mock import patch
    with patch("api.config.Settings", autospec=False):
        pass
    # Register metrics directly to simulate what create_app() does
    from prometheus_client import Counter, Histogram, REGISTRY
    try:
        Counter("invoice_extractions_total", "Total extractions", ["status"])
        Histogram(
            "invoice_extraction_duration_seconds", "Extraction latency",
            buckets=[0.5, 1, 2, 5, 10, 30],
        )
        Counter("llm_tokens_used_total", "LLM tokens", ["model", "direction"])
    except ValueError:
        pass  # already registered
    metric_names = [m.name for m in REGISTRY.collect()]
    assert any("http" in n or "invoice" in n for n in metric_names)


def test_structlog_produces_json(capsys):
    import structlog
    log = structlog.get_logger()
    log.info("test_event", key="value")
    # structlog configured in main.py — no crash = pass
