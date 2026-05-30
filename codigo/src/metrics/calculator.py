from typing import Dict, Any


def compute_metrics(match_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calcula Precisão, Recall e F1 a partir de um resultado de matching.

    TP = número de pares casados
    FP = itens da hipótese sem correspondência na referência (excesso)
    FN = itens da referência sem correspondência na hipótese (ausências)

    Precisão = TP / (TP + FP)  → fração dos itens da IA que são corretos
    Recall   = TP / (TP + FN)  → fração dos itens humanos que a IA encontrou
    F1       = 2 × P × R / (P + R)
    """
    tp = len(match_result["matched"])
    fp = len(match_result["unmatched_hypothesis"])
    fn = len(match_result["unmatched_reference"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }
