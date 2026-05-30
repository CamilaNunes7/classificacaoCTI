from typing import Dict, List, Any

from src.comparator import exact_matcher, fuzzy_matcher, semantic_matcher, bert_scorer
from src.comparator.normalizer import normalize_list
from src.metrics.calculator import compute_metrics

# Os três pares de comparação: (referência, hipótese)
COMPARISON_PAIRS = [
    ("human", "gemini"),
    ("human", "claude"),
    ("gemini", "claude"),
]


def compare_article(
    article_id: str,
    indicators: Dict[str, List[str]],
) -> Dict[str, Any]:
    """
    Executa os quatro métodos de comparação para os três pares de um artigo.

    Parâmetros:
      article_id: identificador do artigo (ex: "article_01")
      indicators: dicionário com chaves "human", "gemini", "claude",
                  cada um mapeando para uma lista de strings brutas.

    Retorna um dict com a estrutura:
    {
      "article_id": str,
      "indicator_counts": {"human": N, "gemini": N, "claude": N},
      "pairs": {
        "human_vs_gemini": {
          "exact":      {matched, unmatched_reference, unmatched_hypothesis, tp, fp, fn, P, R, F1},
          "fuzzy":      {...},
          "semantic":   {...},
          "bertscore":  {...},
        },
        ...
      }
    }
    """
    result: Dict[str, Any] = {
        "article_id": article_id,
        "indicator_counts": {k: len(v) for k, v in indicators.items()},
        "pairs": {},
    }

    for ref_name, hyp_name in COMPARISON_PAIRS:
        # Pula o par se algum dos lados não estiver disponível
        if ref_name not in indicators or hyp_name not in indicators:
            continue

        ref_raw: List[str] = indicators[ref_name]
        hyp_raw: List[str] = indicators[hyp_name]

        # Normalização apenas para comparação exata e fuzzy
        ref_norm = normalize_list(ref_raw)
        hyp_norm = normalize_list(hyp_raw)

        exact_result = exact_matcher.match(ref_norm, hyp_norm)
        fuzzy_result = fuzzy_matcher.match(ref_norm, hyp_norm)
        # Semântico e BERTScore usam texto bruto para preservar qualidade dos embeddings
        semantic_result = semantic_matcher.match(ref_raw, hyp_raw)
        bertscore_result = bert_scorer.match(ref_raw, hyp_raw)

        pair_key = f"{ref_name}_vs_{hyp_name}"
        result["pairs"][pair_key] = {
            "exact":     {**exact_result,     **compute_metrics(exact_result)},
            "fuzzy":     {**fuzzy_result,     **compute_metrics(fuzzy_result)},
            "semantic":  {**semantic_result,  **compute_metrics(semantic_result)},
            "bertscore": {**bertscore_result, **compute_metrics(bertscore_result)},
        }

    return result
