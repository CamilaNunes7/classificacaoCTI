import base64
import logging
from pathlib import Path
from typing import List

import anthropic

from config.settings import ANTHROPIC_API_KEY, CLAUDE_MODEL
from src.extractor.base_extractor import BaseExtractor, parse_json_response
from src.extractor.prompt import EXTRACTION_PROMPT

log = logging.getLogger(__name__)


class ClaudeExtractor(BaseExtractor):
    """Extrator de indicadores CTI usando Anthropic Claude com leitura nativa de PDF."""

    def __init__(self):
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY não configurada. Verifique o arquivo .env.")
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    def extract(self, pdf_path: Path) -> List[str]:
        log.info(f"  [Claude] Enviando PDF: {pdf_path.name}")

        # Lê e codifica o PDF em base64 para envio via document block
        pdf_bytes = pdf_path.read_bytes()
        pdf_b64 = base64.standard_b64encode(pdf_bytes).decode("utf-8")

        response = self.client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": pdf_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": EXTRACTION_PROMPT,
                        },
                    ],
                }
            ],
        )

        raw_text = response.content[0].text
        indicators = parse_json_response(raw_text)
        log.info(f"  [Claude] {len(indicators)} indicadores extraídos de {pdf_path.name}")
        return indicators
