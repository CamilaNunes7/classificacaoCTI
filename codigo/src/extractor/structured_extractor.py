import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List

from google import genai
from google.genai import types

from config.settings import GEMINI_API_KEY, GEMINI_STRUCTURED_MODEL
from src.dataset.schema import AREAS
from src.extractor.structured_prompt import STRUCTURED_EXTRACTION_PROMPT

log = logging.getLogger(__name__)


def _parse_structured_response(text: str) -> List[Dict[str, Any]]:
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        data = json.loads(cleaned)
        raw_list = data.get("indicators", [])
        if not isinstance(raw_list, list):
            return []
        result = []
        for item in raw_list:
            if not isinstance(item, dict) or not item.get("name"):
                continue
            area = item.get("area", "Outros")
            if area not in AREAS:
                area = "Outros"
            result.append({
                "name": str(item.get("name", "")).strip(),
                "area": area,
                "subarea": str(item.get("subarea", "Geral")).strip(),
                "keywords": item.get("keywords", []) if isinstance(item.get("keywords"), list) else [],
                "unit": str(item.get("unit", "N/A")).strip(),
                "excerpt": str(item.get("excerpt", "")).strip(),
                "language": str(item.get("language", "pt")).strip(),
            })
        return result
    except Exception as e:
        log.error(f"Erro ao parsear resposta estruturada: {e}\nTexto: {text[:300]}")
        return []


class StructuredGeminiExtractor:
    """Extrai indicadores CTI com metadados completos usando Gemini 2.5 Pro."""

    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY não configurada.")
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def extract(self, pdf_path: Path) -> List[Dict[str, Any]]:
        log.info(f"  [Gemini-Structured] Enviando PDF: {pdf_path.name}")

        uploaded_file = self.client.files.upload(
            file=str(pdf_path),
            config=types.UploadFileConfig(mime_type="application/pdf"),
        )

        response = self.client.models.generate_content(
            model=GEMINI_STRUCTURED_MODEL,
            contents=[uploaded_file, STRUCTURED_EXTRACTION_PROMPT],
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=8192,
            ),
        )

        try:
            self.client.files.delete(name=uploaded_file.name)
        except Exception:
            pass

        indicators = _parse_structured_response(response.text)
        log.info(f"  [Gemini-Structured] {len(indicators)} indicadores de {pdf_path.name}")
        return indicators

    def extract_with_retry(self, pdf_path: Path, max_retries: int = 3) -> List[Dict[str, Any]]:
        for attempt in range(max_retries):
            try:
                return self.extract(pdf_path)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait = (2 ** attempt) * 10
                log.warning(f"Tentativa {attempt + 1}/{max_retries} falhou: {e}. Aguardando {wait}s…")
                time.sleep(wait)
        return []
