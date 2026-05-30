from pathlib import Path
from typing import List


def parse_annotation_file(filepath: Path) -> List[str]:
    """
    Lê um arquivo TXT com indicadores CTI anotados manualmente.
    Retorna uma lista de strings, uma por linha não vazia.
    Linhas começando com '#' são tratadas como comentários e ignoradas.
    """
    text = filepath.read_text(encoding="utf-8")
    indicators = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    return indicators
