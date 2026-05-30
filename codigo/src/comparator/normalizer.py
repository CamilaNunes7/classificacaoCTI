import re
import unicodedata
from typing import List


def normalize(text: str) -> str:
    """
    Normalização canônica para comparação exata e fuzzy:
    1. Strip de espaços
    2. Remove pontuação final (;.,) — evita falsos negativos por artefatos de formatação
    3. Lowercase
    4. Remove acentos (NFD + descarta categoria Mn)
    5. Colapsa múltiplos espaços em um único
    """
    text = text.strip()
    text = text.rstrip(";.,")
    text = text.strip()
    text = text.lower()
    # NFD decompõe caracteres acentuados: "ç" → "c" + cedilha combinante
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_list(indicators: List[str]) -> List[str]:
    return [normalize(ind) for ind in indicators]
