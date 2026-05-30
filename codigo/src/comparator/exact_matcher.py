from typing import List, Dict, Any


def match(reference: List[str], hypothesis: List[str]) -> Dict[str, Any]:
    """
    Comparação exata após normalização (entrada já deve estar normalizada).

    Retorna:
      matched: lista de (ref_text, hyp_text, score=1.0) para correspondências exatas
      unmatched_reference: itens da referência ausentes na hipótese (falsos negativos)
      unmatched_hypothesis: itens da hipótese ausentes na referência (falsos positivos)
    """
    ref_set = set(reference)
    hyp_set = set(hypothesis)

    tp_texts = ref_set & hyp_set

    return {
        "matched": [(t, t, 1.0) for t in sorted(tp_texts)],
        "unmatched_reference": sorted(ref_set - hyp_set),
        "unmatched_hypothesis": sorted(hyp_set - ref_set),
    }
