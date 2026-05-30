"""
Extrai indicadores não-CT&I dos PDFs em data/input/negative_documents/
e acumula os resultados em data/input/negative_examples/negative_indicators.json.

Uso:
  python extract_negatives.py                  # processa todos os PDFs novos
  python extract_negatives.py --force          # reprocessa mesmo os já extraídos
  python extract_negatives.py --domain saude   # processa apenas um domínio

Estrutura esperada:
  data/input/negative_documents/saude/         *.pdf
  data/input/negative_documents/educacao/      *.pdf
  data/input/negative_documents/meio_ambiente/ *.pdf
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from tqdm import tqdm

from config.settings import BASE_DIR, API_MAX_RETRIES
from src.extractor.negative_extractor import NegativeExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

NEGATIVE_DOCS_DIR = BASE_DIR / "data" / "input" / "negative_documents"
NEGATIVE_OUTPUT = BASE_DIR / "data" / "input" / "negative_examples" / "negative_indicators.json"
DOMAINS = ["saude", "educacao", "meio_ambiente", "economia"]


def load_existing(path: Path) -> list:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return []


def already_extracted(existing: list, pdf_name: str) -> bool:
    return any(item.get("source_pdf") == pdf_name for item in existing)


def save(path: Path, data: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extrai indicadores negativos (não-CT&I) de PDFs")
    parser.add_argument("--force", action="store_true", help="Reprocessa PDFs já extraídos")
    parser.add_argument("--domain", choices=DOMAINS, default=None, help="Processa apenas este domínio (saude, educacao, meio_ambiente, economia)")
    parser.add_argument("--delay", type=int, default=0, metavar="SEG", help="Segundos de pausa extra entre PDFs além do delay entre chunks (padrão: 0)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    domains = [args.domain] if args.domain else DOMAINS

    pdfs = []
    for domain in domains:
        domain_dir = NEGATIVE_DOCS_DIR / domain
        if not domain_dir.exists():
            log.warning(f"Diretório não encontrado: {domain_dir}")
            continue
        for pdf in sorted(domain_dir.glob("*.pdf")):
            pdfs.append((pdf, domain))

    if not pdfs:
        log.error(
            f"Nenhum PDF encontrado em {NEGATIVE_DOCS_DIR}/{{dominio}}/. "
            "Coloque os PDFs nas subpastas saude/, educacao/ ou meio_ambiente/."
        )
        sys.exit(1)

    log.info(f"{len(pdfs)} PDF(s) encontrado(s) nas pastas: {domains}")

    existing = load_existing(NEGATIVE_OUTPUT)
    log.info(f"Indicadores já existentes: {len(existing)}")

    try:
        extractor = NegativeExtractor()
    except ValueError as e:
        log.error(str(e))
        sys.exit(1)

    new_count = 0
    for i, (pdf_path, domain) in enumerate(tqdm(pdfs, desc="Extraindo indicadores negativos")):
        if not args.force and already_extracted(existing, pdf_path.name):
            log.info(f"[{pdf_path.name}] Já extraído — pulando. Use --force para reprocessar.")
            continue

        if args.delay > 0:
            log.info(f"Aguardando {args.delay}s antes de enviar {pdf_path.name}...")
            time.sleep(args.delay)

        try:
            indicators = extractor.extract_with_retry(pdf_path, max_retries=API_MAX_RETRIES)
        except Exception as e:
            log.error(f"[{pdf_path.name}] Falha na extração: {e}")
            continue

        if not indicators:
            log.warning(f"[{pdf_path.name}] Nenhum indicador extraído — entradas anteriores mantidas.")
            continue

        # Só substitui entradas anteriores deste PDF se a nova extração teve sucesso
        existing = [item for item in existing if item.get("source_pdf") != pdf_path.name]
        for text in indicators:
            existing.append({
                "text": text,
                "domain": domain,
                "source_pdf": pdf_path.name,
                "language": "pt",
            })
        new_count += len(indicators)
        log.info(f"[{pdf_path.name}] {len(indicators)} indicadores extraídos → domínio: {domain}")

    save(NEGATIVE_OUTPUT, existing)
    log.info(f"Concluído. +{new_count} novos indicadores. Total: {len(existing)}")
    log.info(f"Arquivo salvo em: {NEGATIVE_OUTPUT}")


if __name__ == "__main__":
    main()
