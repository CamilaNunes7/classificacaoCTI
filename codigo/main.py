"""
Pipeline de Extração e Comparação de Indicadores CTI

Uso:
  python main.py                              # pipeline completo
  python main.py --skip-extraction            # pula extração; usa JSONs existentes
  python main.py --only-report                # só regenera relatórios
  python main.py --articles article_01 ...    # processa apenas artigos específicos
  python main.py --no-gemini                  # pula extração com Gemini
  python main.py --no-claude                  # pula extração com Claude

Estrutura esperada:
  data/input/articles/article_01.pdf
  data/input/annotations/article_01.txt      (mesmo nome base do PDF)
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict

from tqdm import tqdm

from config.settings import (
    INPUT_ARTICLES_DIR,
    INPUT_ANNOTATIONS_DIR,
    OUTPUT_EXTRACTED_DIR,
    OUTPUT_COMPARISON_DIR,
    OUTPUT_REPORTS_DIR,
    OUTPUT_DATASET_DIR,
    API_MAX_RETRIES,
)
from src.parser.txt_parser import parse_annotation_file
from src.comparator.comparison_engine import compare_article
from src.reporter.json_reporter import save_extraction, save_comparison, load_extraction, extraction_cache_valid
from src.reporter.csv_reporter import generate_csv
from src.reporter.html_reporter import generate_html

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline de extração e comparação de indicadores CTI"
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Pula a extração via IA e usa os JSONs já gerados.",
    )
    parser.add_argument(
        "--only-report",
        action="store_true",
        help="Pula extração e comparação; só regenera os relatórios.",
    )
    parser.add_argument(
        "--articles",
        nargs="+",
        default=None,
        metavar="ID",
        help="Processa apenas os artigos com esses IDs (nome base sem extensão).",
    )
    parser.add_argument(
        "--no-gemini",
        action="store_true",
        help="Pula a extração com Gemini.",
    )
    parser.add_argument(
        "--no-claude",
        action="store_true",
        help="Pula a extração com Claude.",
    )
    return parser.parse_args()


def discover_articles(
    articles_dir: Path,
    annotations_dir: Path,
    filter_ids: List[str] = None,
) -> List[Dict]:
    """
    Emparelha PDFs com seus TXTs de anotação pelo nome base.
    Retorna lista de dicts com chaves: id, pdf, txt.
    """
    pdfs = {p.stem: p for p in sorted(articles_dir.glob("*.pdf"))}
    txts = {p.stem: p for p in sorted(annotations_dir.glob("*.txt"))}

    common = sorted(set(pdfs) & set(txts))

    missing_txt = set(pdfs) - set(txts)
    missing_pdf = set(txts) - set(pdfs)

    if missing_txt:
        log.warning(f"PDFs sem anotação TXT correspondente: {sorted(missing_txt)}")
    if missing_pdf:
        log.warning(f"TXTs sem PDF correspondente: {sorted(missing_pdf)}")

    articles = [{"id": stem, "pdf": pdfs[stem], "txt": txts[stem]} for stem in common]

    if filter_ids:
        articles = [a for a in articles if a["id"] in filter_ids]
        if not articles:
            log.error(f"Nenhum artigo encontrado com os IDs: {filter_ids}")

    return articles


def create_output_dirs() -> None:
    for d in [
        OUTPUT_EXTRACTED_DIR / "gemini",
        OUTPUT_EXTRACTED_DIR / "claude",
        OUTPUT_COMPARISON_DIR,
        OUTPUT_REPORTS_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


def run_extraction(articles: List[Dict], args: argparse.Namespace) -> None:
    """Fase 1: extrai indicadores via Gemini e Claude."""
    gemini = None
    claude = None

    if not args.no_gemini:
        from src.extractor.gemini_extractor import GeminiExtractor
        try:
            gemini = GeminiExtractor()
        except ValueError as e:
            log.error(f"Gemini desabilitado: {e}")

    if not args.no_claude:
        from src.extractor.claude_extractor import ClaudeExtractor
        try:
            claude = ClaudeExtractor()
        except ValueError as e:
            log.error(f"Claude desabilitado: {e}")

    for article in tqdm(articles, desc="Extraindo indicadores"):
        aid = article["id"]
        pdf = article["pdf"]

        if gemini:
            out_path = OUTPUT_EXTRACTED_DIR / "gemini" / f"{aid}.json"
            if extraction_cache_valid(out_path):
                log.info(f"[{aid}] Gemini: usando cache ({out_path.name})")
            else:
                try:
                    indicators = gemini.extract_with_retry(pdf, API_MAX_RETRIES)
                    save_extraction(out_path, aid, "gemini", indicators)
                except Exception as e:
                    log.error(f"[{aid}] Gemini falhou: {e}")
                    save_extraction(out_path, aid, "gemini", [], error=str(e))

        if claude:
            out_path = OUTPUT_EXTRACTED_DIR / "claude" / f"{aid}.json"
            if extraction_cache_valid(out_path):
                log.info(f"[{aid}] Claude: usando cache ({out_path.name})")
            else:
                try:
                    indicators = claude.extract_with_retry(pdf, API_MAX_RETRIES)
                    save_extraction(out_path, aid, "claude", indicators)
                except Exception as e:
                    log.error(f"[{aid}] Claude falhou: {e}")
                    save_extraction(out_path, aid, "claude", [], error=str(e))


def run_comparison(articles: List[Dict]) -> List[Dict]:
    """Fase 2: compara indicadores humanos × IA e salva JSONs intermediários."""
    all_results = []

    for article in tqdm(articles, desc="Comparando indicadores"):
        aid = article["id"]

        human_indicators = parse_annotation_file(article["txt"])

        # Só inclui no dict os extratores cujos arquivos de extração existem
        available: Dict[str, List] = {"human": human_indicators}
        for model_name in ("gemini", "claude"):
            json_path = OUTPUT_EXTRACTED_DIR / model_name / f"{aid}.json"
            if json_path.exists():
                available[model_name] = load_extraction(json_path)

        counts_str = ", ".join(f"{k}: {len(v)}" for k, v in available.items())
        log.info(f"[{aid}] Contagens — {counts_str}")

        result = compare_article(aid, available)

        comparison_path = OUTPUT_COMPARISON_DIR / f"{aid}_comparison.json"
        save_comparison(comparison_path, result)
        all_results.append(result)

    return all_results


def load_existing_comparisons(articles: List[Dict]) -> List[Dict]:
    """Carrega comparações já calculadas para regeneração de relatório."""
    import json
    all_results = []
    for article in articles:
        path = OUTPUT_COMPARISON_DIR / f"{article['id']}_comparison.json"
        if path.exists():
            all_results.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            log.warning(f"Comparação não encontrada para: {article['id']}")
    return all_results


def _load_area_metrics():
    """Carrega métricas por área se o dataset enriquecido existir."""
    gold_path = OUTPUT_DATASET_DIR / "enriched_gold.json"
    ai_path = OUTPUT_DATASET_DIR / "enriched_ai_gemini.json"
    if not gold_path.exists() or not ai_path.exists():
        return None
    try:
        from src.dataset.store import load_dataset
        from src.metrics.area_metrics import compute_global_area_metrics
        gold = load_dataset(gold_path)
        ai = load_dataset(ai_path)
        results = compute_global_area_metrics(gold, ai)

        gold_totals = {}
        ai_totals = {}
        for ind in gold:
            gold_totals[ind.area] = gold_totals.get(ind.area, 0) + 1
        for ind in ai:
            ai_totals[ind.area] = ai_totals.get(ind.area, 0) + 1

        area_metrics = sorted(
            [
                {
                    "area": area,
                    "ref_count": gold_totals.get(area, 0),
                    "hyp_count": ai_totals.get(area, 0),
                    **m,
                }
                for area, m in results["macro_by_area"].items()
            ],
            key=lambda x: -x["f1"],
        )
        log.info(f"Métricas por área carregadas: {len(area_metrics)} áreas")
        return area_metrics
    except Exception as e:
        log.warning(f"Não foi possível carregar métricas por área: {e}")
        return None


def main() -> None:
    args = parse_args()
    create_output_dirs()

    articles = discover_articles(
        INPUT_ARTICLES_DIR,
        INPUT_ANNOTATIONS_DIR,
        filter_ids=args.articles,
    )

    if not articles:
        log.error(
            "Nenhum artigo encontrado. Verifique se há arquivos .pdf em "
            f"'{INPUT_ARTICLES_DIR}' e .txt em '{INPUT_ANNOTATIONS_DIR}' "
            "com nomes correspondentes."
        )
        sys.exit(1)

    log.info(f"Artigos encontrados: {len(articles)}")

    # ── Fase 1: Extração ──────────────────────────────────────────────────────
    if not args.skip_extraction and not args.only_report:
        log.info("=== Fase 1: Extração via IA ===")
        run_extraction(articles, args)
    else:
        log.info("Fase 1 (extração) ignorada.")

    # ── Fase 2: Comparação ────────────────────────────────────────────────────
    if not args.only_report:
        log.info("=== Fase 2: Comparação de indicadores ===")
        all_results = run_comparison(articles)
    else:
        log.info("Fase 2 (comparação) ignorada. Carregando resultados existentes...")
        all_results = load_existing_comparisons(articles)

    if not all_results:
        log.error("Nenhum resultado de comparação disponível para gerar relatório.")
        sys.exit(1)

    # ── Fase 3: Relatórios ────────────────────────────────────────────────────
    log.info("=== Fase 3: Gerando relatórios ===")
    generate_csv(all_results, OUTPUT_REPORTS_DIR / "metrics_summary.csv")

    # Inclui métricas por área no HTML se o dataset enriquecido estiver disponível
    area_metrics = _load_area_metrics()
    generate_html(all_results, OUTPUT_REPORTS_DIR / "report.html", area_metrics=area_metrics)

    log.info("Pipeline concluído.")
    log.info(f"  CSV  → {OUTPUT_REPORTS_DIR / 'metrics_summary.csv'}")
    log.info(f"  HTML → {OUTPUT_REPORTS_DIR / 'report.html'}")


if __name__ == "__main__":
    main()
