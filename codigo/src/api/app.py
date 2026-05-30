import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from config.settings import (
    GEMINI_API_KEY,
    INPUT_ANNOTATIONS_DIR,
    INPUT_ARTICLES_DIR,
    OUTPUT_COMPARISON_DIR,
    OUTPUT_DATASET_DIR,
    OUTPUT_EXTRACTED_DIR,
    OUTPUT_REPORTS_DIR,
)
from src.api.models import (
    AreaMetricRow,
    AreaMetricsOut,
    ArticleSummary,
    DatasetPage,
    DatasetStats,
    EnrichStatus,
    ExtractionOut,
    IndicatorOut,
    AreaStats,
    SubareaCount,
)
from src.dataset.schema import AREAS, TAXONOMY
from src.dataset.store import dataset_exists, load_dataset, save_dataset

log = logging.getLogger(__name__)

GOLD_PATH = OUTPUT_DATASET_DIR / "enriched_gold.json"
AI_PATH = OUTPUT_DATASET_DIR / "enriched_ai_gemini.json"

app = FastAPI(
    title="CTI Indicators Dataset API",
    description=(
        "API REST para o dataset anotado de indicadores de Ciência, Tecnologia e Inovação (CT&I). "
        "Permite consultar o dataset estruturado, extrair indicadores de novos PDFs com Gemini 2.5 Pro, "
        "e acessar métricas de avaliação globais e por área temática."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Raiz ──────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, tags=["info"])
def root():
    return """<!DOCTYPE html><html lang="pt-BR"><head><meta charset="UTF-8">
    <title>CTI Dataset API</title>
    <style>body{font-family:Arial,sans-serif;max-width:800px;margin:40px auto;color:#222}
    h1{color:#1a237e}a{color:#3f51b5}li{margin:6px 0}</style></head>
    <body>
    <h1>CTI Indicators Dataset API</h1>
    <p>Dataset anotado de indicadores de Ciência, Tecnologia e Inovação (CT&I)</p>
    <ul>
      <li><a href="/docs">Documentação interativa (Swagger UI)</a></li>
      <li><a href="/redoc">Documentação alternativa (ReDoc)</a></li>
      <li><a href="/dataset">GET /dataset</a> — dataset completo com filtros</li>
      <li><a href="/dataset/stats">GET /dataset/stats</a> — estatísticas por área</li>
      <li><a href="/taxonomy">GET /taxonomy</a> — taxonomia de áreas</li>
      <li><a href="/articles">GET /articles</a> — artigos do corpus</li>
      <li><a href="/metrics">GET /metrics</a> — métricas globais de comparação</li>
      <li><a href="/metrics/by-area">GET /metrics/by-area</a> — métricas por área temática</li>
      <li><a href="/report">GET /report</a> — relatório HTML completo</li>
      <li><strong>POST /extract</strong> — extrai indicadores de um PDF (Gemini 2.5 Pro)</li>
      <li><strong>POST /dataset/enrich</strong> — constrói o dataset enriquecido</li>
    </ul>
    </body></html>"""


@app.get("/taxonomy", tags=["info"])
def get_taxonomy():
    """Retorna a taxonomia completa de áreas e subáreas."""
    return {
        "areas": [
            {"area": area, "subareas": subs}
            for area, subs in TAXONOMY.items()
        ]
    }


# ── Dataset ───────────────────────────────────────────────────────────────────

@app.get("/dataset", response_model=DatasetPage, tags=["dataset"])
def get_dataset(
    area: Optional[str] = Query(None, description="Filtrar por área temática"),
    source_article: Optional[str] = Query(None, description="Filtrar por artigo (ex: artigo1)"),
    language: Optional[str] = Query(None, description="Idioma: 'pt' ou 'en'"),
    annotated_by: Optional[str] = Query(None, description="Anotador: 'human' ou 'ai'"),
    limit: int = Query(100, ge=1, le=1000, description="Itens por página"),
    offset: int = Query(0, ge=0, description="Deslocamento inicial"),
):
    """Retorna o dataset estruturado de indicadores CTI com filtros opcionais."""
    if not dataset_exists(GOLD_PATH):
        raise HTTPException(
            status_code=404,
            detail="Dataset não encontrado. Execute POST /dataset/enrich para construí-lo.",
        )

    indicators = load_dataset(GOLD_PATH)

    if area:
        indicators = [i for i in indicators if i.area == area]
    if source_article:
        indicators = [i for i in indicators if i.source_article == source_article]
    if language:
        indicators = [i for i in indicators if i.language == language]
    if annotated_by:
        indicators = [i for i in indicators if i.annotated_by == annotated_by]

    total = len(indicators)
    page = indicators[offset : offset + limit]

    return DatasetPage(
        total=total,
        offset=offset,
        limit=limit,
        indicators=[IndicatorOut(**i.to_dict()) for i in page],
    )


@app.get("/dataset/stats", response_model=DatasetStats, tags=["dataset"])
def get_dataset_stats():
    """Retorna estatísticas do dataset por área, subárea, idioma e anotador."""
    if not dataset_exists(GOLD_PATH):
        return DatasetStats(
            total_indicators=0,
            total_articles=0,
            enriched=False,
            areas=[],
            by_annotator={},
            by_language={},
        )

    indicators = load_dataset(GOLD_PATH)

    area_map: Dict[str, Dict[str, int]] = {}
    annotator_counts: Dict[str, int] = {}
    language_counts: Dict[str, int] = {}
    articles = set()

    for ind in indicators:
        area_map.setdefault(ind.area, {})
        area_map[ind.area][ind.subarea] = area_map[ind.area].get(ind.subarea, 0) + 1
        annotator_counts[ind.annotated_by] = annotator_counts.get(ind.annotated_by, 0) + 1
        language_counts[ind.language] = language_counts.get(ind.language, 0) + 1
        articles.add(ind.source_article)

    areas = sorted(
        [
            AreaStats(
                area=area,
                count=sum(subs.values()),
                subareas=[
                    SubareaCount(subarea=sub, count=cnt)
                    for sub, cnt in sorted(subs.items(), key=lambda x: -x[1])
                ],
            )
            for area, subs in area_map.items()
        ],
        key=lambda x: -x.count,
    )

    return DatasetStats(
        total_indicators=len(indicators),
        total_articles=len(articles),
        enriched=True,
        areas=areas,
        by_annotator=annotator_counts,
        by_language=language_counts,
    )


@app.post("/dataset/enrich", response_model=EnrichStatus, tags=["dataset"])
def enrich_dataset(
    background_tasks: BackgroundTasks,
    include_ai: bool = Query(
        False,
        description="Também classifica as extrações do Gemini (útil para métricas por área)",
    ),
):
    """
    Constrói o dataset estruturado enriquecendo o padrão ouro com metadados
    (área, subárea, keywords, unidade) via Gemini 2.5 Pro.

    O processamento roda em background — consulte GET /dataset/stats para acompanhar.
    """
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY não configurada.")
    background_tasks.add_task(_run_enrichment, include_ai=include_ai)
    return EnrichStatus(
        status="started",
        message=(
            "Enriquecimento iniciado em background. "
            "Consulte GET /dataset/stats para verificar quando concluir."
        ),
    )


def _run_enrichment(include_ai: bool = False) -> None:
    from src.parser.txt_parser import parse_annotation_file
    from src.dataset.enricher import DatasetEnricher

    log.info("=== Enriquecimento do dataset iniciado ===")
    OUTPUT_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    enricher = DatasetEnricher()

    # ── Padrão ouro (anotações humanas) ──────────────────────────────────────
    all_gold = []
    for txt_file in sorted(INPUT_ANNOTATIONS_DIR.glob("*.txt")):
        article_id = txt_file.stem
        names = parse_annotation_file(txt_file)
        log.info(f"[{article_id}] Enriquecendo {len(names)} indicadores humanos…")
        all_gold.extend(enricher.enrich_article(article_id, names, annotated_by="human"))

    save_dataset(all_gold, GOLD_PATH, {"type": "gold_standard", "annotated_by": "human"})
    log.info(f"Dataset padrão ouro salvo: {len(all_gold)} indicadores → {GOLD_PATH}")

    # ── Extrações da IA (opcional) ────────────────────────────────────────────
    if include_ai:
        all_ai = []
        gemini_dir = OUTPUT_EXTRACTED_DIR / "gemini"
        for gf in sorted(gemini_dir.glob("*.json")):
            article_id = gf.stem
            data = json.loads(gf.read_text(encoding="utf-8"))
            names = data.get("indicators", [])
            if names:
                log.info(f"[{article_id}] Classificando {len(names)} indicadores Gemini…")
                all_ai.extend(
                    enricher.enrich_article(article_id, names, annotated_by="ai")
                )

        save_dataset(all_ai, AI_PATH, {"type": "ai_extracted", "model": "gemini"})
        log.info(f"Dataset AI salvo: {len(all_ai)} indicadores → {AI_PATH}")

    log.info("=== Enriquecimento concluído ===")


# ── Artigos ───────────────────────────────────────────────────────────────────

@app.get("/articles", tags=["corpus"])
def get_articles():
    """Lista todos os artigos do corpus com contagem de indicadores por fonte."""
    articles = []
    for txt_file in sorted(INPUT_ANNOTATIONS_DIR.glob("*.txt")):
        article_id = txt_file.stem
        lines = [
            l.strip()
            for l in txt_file.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.strip().startswith("#")
        ]
        gemini_path = OUTPUT_EXTRACTED_DIR / "gemini" / f"{article_id}.json"
        gemini_count: Optional[int] = None
        if gemini_path.exists():
            try:
                d = json.loads(gemini_path.read_text(encoding="utf-8"))
                gemini_count = d.get("indicator_count", len(d.get("indicators", [])))
            except Exception:
                pass

        articles.append(
            ArticleSummary(
                id=article_id,
                human_count=len(lines),
                gemini_count=gemini_count,
                has_pdf=(INPUT_ARTICLES_DIR / f"{article_id}.pdf").exists(),
                has_comparison=(OUTPUT_COMPARISON_DIR / f"{article_id}_comparison.json").exists(),
            )
        )
    return {"total": len(articles), "articles": articles}


# ── Extração ──────────────────────────────────────────────────────────────────

@app.post("/extract", response_model=ExtractionOut, tags=["extraction"])
async def extract_from_pdf(file: UploadFile = File(...)):
    """
    Extrai indicadores CTI estruturados de um PDF usando Gemini 2.5 Pro.

    Retorna nome, área, subárea, keywords, unidade e trecho original para cada indicador.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Apenas arquivos PDF são aceitos.")
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY não configurada.")

    content = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        from src.extractor.structured_extractor import StructuredGeminiExtractor

        extractor = StructuredGeminiExtractor()
        indicators = extractor.extract_with_retry(tmp_path)
    except Exception as e:
        log.error(f"Erro na extração de {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Erro na extração: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)

    return ExtractionOut(
        filename=file.filename,
        model="gemini-2.5-pro",
        indicator_count=len(indicators),
        indicators=indicators,
    )


# ── Métricas ──────────────────────────────────────────────────────────────────

@app.get("/metrics/by-area", response_model=AreaMetricsOut, tags=["metrics"])
def get_area_metrics():
    """
    Retorna métricas de Precisão/Recall/F1 por área temática (Fuzzy, Humano × Gemini).
    Requer que o dataset tenha sido enriquecido com POST /dataset/enrich?include_ai=true.
    """
    if not dataset_exists(GOLD_PATH):
        return AreaMetricsOut(
            enriched=False,
            message="Execute POST /dataset/enrich para construir o dataset enriquecido.",
            macro_by_area=[],
        )
    if not dataset_exists(AI_PATH):
        return AreaMetricsOut(
            enriched=False,
            message="Execute POST /dataset/enrich?include_ai=true para classificar as extrações AI.",
            macro_by_area=[],
        )

    from src.metrics.area_metrics import compute_global_area_metrics

    gold = load_dataset(GOLD_PATH)
    ai = load_dataset(AI_PATH)
    results = compute_global_area_metrics(gold, ai)

    gold_area_totals: Dict[str, int] = {}
    ai_area_totals: Dict[str, int] = {}
    for ind in gold:
        gold_area_totals[ind.area] = gold_area_totals.get(ind.area, 0) + 1
    for ind in ai:
        ai_area_totals[ind.area] = ai_area_totals.get(ind.area, 0) + 1

    macro_list = sorted(
        [
            AreaMetricRow(
                area=area,
                ref_count=gold_area_totals.get(area, 0),
                hyp_count=ai_area_totals.get(area, 0),
                precision=m["precision"],
                recall=m["recall"],
                f1=m["f1"],
            )
            for area, m in results["macro_by_area"].items()
        ],
        key=lambda x: -x.f1,
    )

    return AreaMetricsOut(
        enriched=True,
        macro_by_area=macro_list,
        per_article=results["per_article"],
    )


@app.get("/metrics", tags=["metrics"])
def get_metrics():
    """Retorna métricas globais de comparação para todos os artigos e métodos."""
    comparison_files = sorted(OUTPUT_COMPARISON_DIR.glob("*_comparison.json"))
    if not comparison_files:
        raise HTTPException(
            status_code=404,
            detail="Nenhum resultado de comparação encontrado. Execute python main.py primeiro.",
        )

    all_results = [
        json.loads(cf.read_text(encoding="utf-8")) for cf in comparison_files
    ]

    METHODS = ["exact", "fuzzy", "semantic", "bertscore"]
    sums: Dict[str, Dict[str, Dict[str, float]]] = {}
    counts: Dict[str, Dict[str, int]] = {}

    for result in all_results:
        for pair_key, pair_data in result.get("pairs", {}).items():
            sums.setdefault(pair_key, {})
            counts.setdefault(pair_key, {})
            for method in METHODS:
                sums[pair_key].setdefault(method, {"precision": 0.0, "recall": 0.0, "f1": 0.0})
                counts[pair_key].setdefault(method, 0)
                m = pair_data.get(method, {})
                sums[pair_key][method]["precision"] += m.get("precision", 0.0)
                sums[pair_key][method]["recall"] += m.get("recall", 0.0)
                sums[pair_key][method]["f1"] += m.get("f1", 0.0)
                counts[pair_key][method] += 1

    aggregates: Dict[str, Any] = {}
    for pair_key, methods in sums.items():
        aggregates[pair_key] = {}
        for method, totals in methods.items():
            n = counts[pair_key][method] or 1
            aggregates[pair_key][method] = {
                "precision": round(totals["precision"] / n, 4),
                "recall": round(totals["recall"] / n, 4),
                "f1": round(totals["f1"] / n, 4),
            }

    return {
        "total_articles": len(all_results),
        "aggregates": aggregates,
        "articles": all_results,
    }


@app.get("/metrics/{article_id}", tags=["metrics"])
def get_article_metrics(article_id: str):
    """Retorna métricas de comparação para um artigo específico."""
    path = OUTPUT_COMPARISON_DIR / f"{article_id}_comparison.json"
    if not path.exists():
        raise HTTPException(
            status_code=404, detail=f"Artigo '{article_id}' não encontrado."
        )
    return json.loads(path.read_text(encoding="utf-8"))


# ── Relatório ─────────────────────────────────────────────────────────────────

@app.get("/report", response_class=HTMLResponse, tags=["report"])
def get_report():
    """Serve o relatório HTML completo gerado pelo pipeline."""
    report_path = OUTPUT_REPORTS_DIR / "report.html"
    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Relatório não encontrado. Execute: python main.py --only-report",
        )
    return HTMLResponse(content=report_path.read_text(encoding="utf-8"))
