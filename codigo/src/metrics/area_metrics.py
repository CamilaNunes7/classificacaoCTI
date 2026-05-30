import logging
from collections import defaultdict
from typing import Any, Dict, List

from src.comparator import fuzzy_matcher
from src.comparator.normalizer import normalize_list
from src.dataset.schema import Indicator
from src.metrics.calculator import compute_metrics

log = logging.getLogger(__name__)


def _group_names_by_area(indicators: List[Indicator], article_id: str) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = defaultdict(list)
    for ind in indicators:
        if ind.source_article == article_id:
            groups[ind.area].append(ind.name)
    return dict(groups)


def compute_area_metrics_for_article(
    article_id: str,
    gold: List[Indicator],
    ai: List[Indicator],
) -> Dict[str, Any]:
    """Calcula P/R/F1 por área para um artigo (usando fuzzy matching)."""
    gold_by_area = _group_names_by_area(gold, article_id)
    ai_by_area = _group_names_by_area(ai, article_id)

    all_areas = sorted(set(gold_by_area) | set(ai_by_area))
    area_results: Dict[str, Any] = {}

    for area in all_areas:
        ref = normalize_list(gold_by_area.get(area, []))
        hyp = normalize_list(ai_by_area.get(area, []))

        if not ref and not hyp:
            continue

        match_result = fuzzy_matcher.match(ref, hyp)
        metrics = compute_metrics(match_result)
        area_results[area] = {
            "ref_count": len(ref),
            "hyp_count": len(hyp),
            **metrics,
        }

    return area_results


def compute_global_area_metrics(
    gold: List[Indicator],
    ai: List[Indicator],
) -> Dict[str, Any]:
    """Calcula métricas por área agregadas em macro-média sobre todos os artigos."""
    articles = sorted({i.source_article for i in gold})
    area_sums: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "count": 0}
    )
    per_article: Dict[str, Any] = {}

    for article_id in articles:
        article_areas = compute_area_metrics_for_article(article_id, gold, ai)
        per_article[article_id] = article_areas
        for area, m in article_areas.items():
            area_sums[area]["precision"] += m["precision"]
            area_sums[area]["recall"] += m["recall"]
            area_sums[area]["f1"] += m["f1"]
            area_sums[area]["count"] += 1

    macro: Dict[str, Any] = {}
    for area, sums in area_sums.items():
        n = sums["count"] or 1
        macro[area] = {
            "precision": round(sums["precision"] / n, 4),
            "recall": round(sums["recall"] / n, 4),
            "f1": round(sums["f1"] / n, 4),
            "article_count": int(sums["count"]),
        }

    return {"macro_by_area": macro, "per_article": per_article}
