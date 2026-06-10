import os


def get_ollama_llm(model: str, temperature: float = 0):
    from langchain_ollama import OllamaLLM

    return OllamaLLM(
        model=model,
        temperature=temperature,
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
