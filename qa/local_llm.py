import requests
import json

class OllamaLLM:
    def __init__(self, model="mistral", base_url="http://localhost:11434"):
        self.model = model
        self.url = f"{base_url}/api/generate"

    def generate(self, prompt, temperature=0.0):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "stream": False
        }

        response = requests.post(self.url, json=payload)

        if response.status_code != 200:
            raise RuntimeError(f"Ollama error: {response.text}")

        return response.json()["response"]
