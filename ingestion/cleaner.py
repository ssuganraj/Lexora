import re

def clean_text(text: str) -> str:
    """
    Basic text cleaning:
    - Remove excessive newlines
    - Strip leading/trailing spaces
    """
    text = re.sub(r'\n{2,}', '\n\n', text)
    return text.strip()
