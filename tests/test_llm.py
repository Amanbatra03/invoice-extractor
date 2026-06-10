from rag.llm import get_ollama_llm


def test_default_base_url(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    llm = get_ollama_llm("llama3.2:3b")
    assert llm.base_url == "http://localhost:11434"
    assert llm.model == "llama3.2:3b"
    assert llm.temperature == 0


def test_base_url_from_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama:11434")
    llm = get_ollama_llm("llama3.2:3b")
    assert llm.base_url == "http://ollama:11434"


def test_format_schema_passed_through(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    llm = get_ollama_llm("llama3.2:3b", format_schema=schema)
    assert llm.format == "json"


def test_num_ctx_default():
    llm = get_ollama_llm("llama3.2:3b")
    assert llm.num_ctx == 8192
