import logging
from pathlib import Path
from typing import List

from google import genai
from google.genai import types

from config.settings import GEMINI_API_KEY, GEMINI_MODEL
from src.extractor.base_extractor import BaseExtractor, parse_json_response
from src.extractor.prompt import EXTRACTION_PROMPT

log = logging.getLogger(__name__)


class GeminiExtractor(BaseExtractor):
    """Extrator de indicadores CTI usando Google Gemini com leitura nativa de PDF."""

    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY não configurada. Verifique o arquivo .env.")
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def extract(self, pdf_path: Path) -> List[str]:
        log.info(f"  [Gemini] Enviando PDF: {pdf_path.name}")

        uploaded_file = self.client.files.upload(
            file=str(pdf_path),
            config=types.UploadFileConfig(mime_type="application/pdf"),
        )
        log.debug(f"  [Gemini] Arquivo enviado: {uploaded_file.uri}")

        response = self.client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[uploaded_file, EXTRACTION_PROMPT],
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=4096,
            ),
        )

        try:
            self.client.files.delete(name=uploaded_file.name)
        except Exception:
            pass

        raw_text = response.text
        indicators = parse_json_response(raw_text)
        log.info(f"  [Gemini] {len(indicators)} indicadores extraídos de {pdf_path.name}")
        return indicators
