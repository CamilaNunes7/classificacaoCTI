from typing import List, Dict, Any

from rapidfuzz import fuzz, process

from config.settings import FUZZY_THRESHOLD


def match(
    reference: List[str],
    hypothesis: List[str],
    threshold: float = FUZZY_THRESHOLD,
) -> Dict[str, Any]:
    """
    Matching fuzzy usando rapidfuzz.fuzz.WRatio.
    WRatio lida com: variações de ordem de tokens, correspondências parciais,
    abreviações (ex: 'P&D' ≈ 'pesquisa e desenvolvimento').

    Score: 0–100. Threshold padrão: 80.
    Matching guloso: para cada item da referência, encontra o melhor item
    ainda não casado na hipótese acima do threshold.
    """
    matched = []
    matched_hyp_set: set = set()
    unmatched_ref = []

    for ref_text in reference:
        remaining = [h for h in hypothesis if h not in matched_hyp_set]
        if not remaining:
            unmatched_ref.append(ref_text)
            continue

        result = process.extractOne(
            ref_text,
            remaining,
            scorer=fuzz.WRatio,
            score_cutoff=threshold,
        )

        if result is not None:
            best_text, best_score, _ = result
            matched.append((ref_text, best_text, round(best_score / 100.0, 4)))
            matched_hyp_set.add(best_text)
        else:
            unmatched_ref.append(ref_text)

    unmatched_hyp = [h for h in hypothesis if h not in matched_hyp_set]

    return {
        "matched": matched,
        "unmatched_reference": unmatched_ref,
        "unmatched_hypothesis": unmatched_hyp,
    }
