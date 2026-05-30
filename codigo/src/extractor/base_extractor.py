import json
import logging
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

log = logging.getLogger(__name__)


class BaseExtractor(ABC):
    """Classe base abstrata para extratores de indicadores CTI via LLM."""

    @abstractmethod
    def extract(self, pdf_path: Path) -> List[str]:
        """Extrai indicadores CTI de um PDF. Retorna lista de strings brutas."""
        ...

    def extract_with_retry(self, pdf_path: Path, max_retries: int = 3) -> List[str]:
        """Envolve extract() com lógica de retry com backoff exponencial."""
        for attempt in range(max_retries):
            try:
                return self.extract(pdf_path)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait = (2 ** attempt) * 60
                log.warning(
                    f"Tentativa {attempt + 1}/{max_retries} falhou: {e}. "
                    f"Tentando novamente em {wait}s..."
                )
                time.sleep(wait)
        return []


def parse_json_response(text: str) -> List[str]:
    """
    Parseia a resposta JSON do LLM para extrair a lista de indicadores.
    Lida com markdown fences e whitespace extras defensivamente.
    """
    cleaned = text.strip()

    # Remove markdown code fences se presentes (```json ... ``` ou ``` ... ```)
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
        indicators = data.get("indicators", [])
        if not isinstance(indicators, list):
            log.warning("Campo 'indicators' não é uma lista. Retornando vazio.")
            return []
        return [str(ind).strip() for ind in indicators if str(ind).strip()]
    except json.JSONDecodeError as e:
        log.error(f"Falha ao parsear JSON da resposta do LLM: {e}\nResposta recebida:\n{text[:500]}")
        return []
