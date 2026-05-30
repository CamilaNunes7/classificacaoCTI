import logging
import tempfile
import time
from pathlib import Path
from typing import List

from google import genai
from google.genai import types
from pypdf import PdfReader, PdfWriter

from config.settings import GEMINI_API_KEY, GEMINI_STRUCTURED_MODEL as GEMINI_MODEL
from src.extractor.base_extractor import parse_json_response
from src.extractor.negative_prompt import NEGATIVE_EXTRACTION_PROMPT

log = logging.getLogger(__name__)

CHUNK_SIZE = 20      # páginas por chunk
CHUNK_DELAY = 60     # segundos entre chunks


def _split_pdf(pdf_path: Path, chunk_size: int) -> List[Path]:
    """Divide o PDF em arquivos temporários de chunk_size páginas cada."""
    reader = PdfReader(str(pdf_path))
    total = len(reader.pages)
    chunks = []

    for start in range(0, total, chunk_size):
        writer = PdfWriter()
        for page in reader.pages[start: start + chunk_size]:
            writer.add_page(page)

        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        writer.write(tmp)
        tmp.close()
        chunks.append(Path(tmp.name))

    log.info(f"  PDF dividido em {len(chunks)} chunk(s) de até {chunk_size} páginas ({total} páginas total)")
    return chunks


class NegativeExtractor:
    """Extrai indicadores não-CT&I de documentos públicos (saúde, educação, meio ambiente, economia).

    PDFs grandes são divididos em chunks para evitar limite de TPM da API.
    """

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_delay: int = CHUNK_DELAY):
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY não configurada. Verifique o arquivo .env.")
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.chunk_size = chunk_size
        self.chunk_delay = chunk_delay

    def _extract_chunk(self, chunk_path: Path, chunk_label: str) -> List[str]:
        """Envia um chunk para a API e retorna os indicadores extraídos."""
        log.info(f"    Enviando {chunk_label}...")
        uploaded = self.client.files.upload(
            file=str(chunk_path),
            config=types.UploadFileConfig(mime_type="application/pdf"),
        )
        try:
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[uploaded, NEGATIVE_EXTRACTION_PROMPT],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=16384,
                ),
            )
            indicators = parse_json_response(response.text)
            log.info(f"    {chunk_label}: {len(indicators)} indicadores")
            return indicators
        finally:
            try:
                self.client.files.delete(name=uploaded.name)
            except Exception:
                pass

    def extract(self, pdf_path: Path) -> List[str]:
        chunks = _split_pdf(pdf_path, self.chunk_size)
        all_indicators: List[str] = []

        try:
            for i, chunk_path in enumerate(chunks):
                label = f"chunk {i + 1}/{len(chunks)}"
                try:
                    indicators = self._extract_chunk(chunk_path, label)
                    all_indicators.extend(indicators)
                except Exception as e:
                    log.error(f"    Falha no {label}: {e}")

                if i < len(chunks) - 1:
                    log.info(f"    Aguardando {self.chunk_delay}s antes do próximo chunk...")
                    time.sleep(self.chunk_delay)
        finally:
            for chunk_path in chunks:
                chunk_path.unlink(missing_ok=True)

        # Remove duplicatas mantendo a ordem
        seen = set()
        unique = []
        for ind in all_indicators:
            if ind not in seen:
                seen.add(ind)
                unique.append(ind)

        log.info(f"  [NegativeExtractor] {len(unique)} indicadores únicos de {pdf_path.name}")
        return unique

    def extract_with_retry(self, pdf_path: Path, max_retries: int = 3) -> List[str]:
        """Mantém compatibilidade com a interface BaseExtractor."""
        import time as _time
        for attempt in range(max_retries):
            try:
                return self.extract(pdf_path)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait = (2 ** attempt) * 60
                log.warning(f"Tentativa {attempt + 1}/{max_retries} falhou: {e}. Tentando em {wait}s...")
                _time.sleep(wait)
        return []
