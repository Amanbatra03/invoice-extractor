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
