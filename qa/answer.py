# qa/answer.py

import requests


class OllamaLLM:
    def __init__(self, model: str = "mistral:latest"):
        self.url = "http://localhost:11434/api/generate"
        self.model = model

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        try:
            response = requests.post(
                self.url,
                json=payload,
                timeout=120  # prevent hanging
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama connection failed: {e}")

        if response.status_code != 200:
            raise RuntimeError(
                f"Ollama error {response.status_code}: {response.text}"
            )

        return response.json().get("response", "")
