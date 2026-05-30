import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

log = logging.getLogger(__name__)

METHODS = ["exact", "fuzzy", "semantic", "bertscore"]
PAIRS = ["human_vs_gemini", "human_vs_claude", "gemini_vs_claude"]


def generate_csv(all_results: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Gera um CSV com uma linha por (artigo × par × método).
    Adiciona linhas de agregação (macro-média) ao final.
    """
    rows = []

    for result in all_results:
        article_id = result["article_id"]
        counts = result.get("indicator_counts", {})

        for pair_key, pair_data in result["pairs"].items():
            ref_name, hyp_name = pair_key.split("_vs_")
            for method in METHODS:
                m = pair_data.get(method, {})
                rows.append({
                    "article_id": article_id,
                    "pair": pair_key,
                    "method": method,
                    "ref_count": counts.get(ref_name, 0),
                    "hyp_count": counts.get(hyp_name, 0),
                    "tp": m.get("tp", 0),
                    "fp": m.get("fp", 0),
                    "fn": m.get("fn", 0),
                    "precision": m.get("precision", 0.0),
                    "recall": m.get("recall", 0.0),
                    "f1": m.get("f1", 0.0),
                })

    df = pd.DataFrame(rows)

    # Linhas de agregação: macro-média por (par × método)
    agg_rows = []
    for pair_key in PAIRS:
        for method in METHODS:
            subset = df[(df["pair"] == pair_key) & (df["method"] == method)]
            if subset.empty:
                continue
            agg_rows.append({
                "article_id": "AGGREGATE",
                "pair": pair_key,
                "method": method,
                "ref_count": None,
                "hyp_count": None,
                "tp": None,
                "fp": None,
                "fn": None,
                "precision": round(subset["precision"].mean(), 4),
                "recall": round(subset["recall"].mean(), 4),
                "f1": round(subset["f1"].mean(), 4),
            })

    agg_df = pd.DataFrame(agg_rows)
    final_df = pd.concat([df, agg_df], ignore_index=True)

    final_df.to_csv(output_path, index=False, encoding="utf-8")
    log.info(f"CSV salvo em: {output_path}")
