import os


def get_ollama_llm(model: str, temperature: float = 0, format_schema: dict | None = None,
                   num_ctx: int = 8192):
    from langchain_ollama import OllamaLLM

    kwargs = dict(
        model=model,
        temperature=temperature,
        num_ctx=num_ctx,   # Ollama's default 2048 silently truncates long invoices
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    if format_schema is not None:
        kwargs["format"] = "json"
    return OllamaLLM(**kwargs)
