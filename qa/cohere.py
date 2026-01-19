import cohere
from qa.prompt import build_prompt

class CohereLLM:
    def __init__(self, api_key: str):
        # Initialize the Cohere client with your free API key
        self.client = cohere.Client(api_key)

    def generate(self, prompt: str, model: str = "command-a-03-2025", max_tokens: int = 256, temperature: float = 0.0):
        """
        Generate an answer using Cohere's chat API version supported by your SDK.
        """
        response = self.client.chat(
            model=model,
            # The older SDK expects a single "message" string
            message=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )

        # The response may return .text depending on the SDK version
        # Try response.text if .message doesn't exist
        if hasattr(response, "message") and hasattr(response.message, "content"):
            # Newer structured responses
            return "".join([c["text"] for c in response.message.content])
        elif hasattr(response, "text"):
            # Older simpler response
            return response.text
        else:
            return str(response)
