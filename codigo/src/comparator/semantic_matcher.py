import logging
from typing import List, Dict, Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import EMBEDDING_MODEL, SEMANTIC_THRESHOLD

log = logging.getLogger(__name__)

# Singleton: modelo carregado uma vez e reutilizado em todas as comparações
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        log.info(f"Carregando modelo de embeddings: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def match(
    reference: List[str],
    hypothesis: List[str],
    threshold: float = SEMANTIC_THRESHOLD,
) -> Dict[str, Any]:
    """
    Matching semântico usando embeddings de sentenças e similaridade de cosseno.

    IMPORTANTE: recebe texto bruto (sem normalização de acentos/lowercase),
    pois o modelo foi treinado com texto natural e produz embeddings melhores
    com texto preservado.

    Estratégia:
    - Embeda todos os indicadores de referência e hipótese.
    - Constrói matriz de similaridade (|ref| × |hyp|).
    - Matching guloso: para cada item da referência, encontra o melhor item
      ainda não casado na hipótese com similaridade ≥ threshold.

    Retorna:
      matched: lista de (ref_text, hyp_text, score) para pares acima do threshold
      unmatched_reference: itens sem correspondência (falsos negativos)
      unmatched_hypothesis: itens extras na hipótese (falsos positivos)
    """
    if not reference or not hypothesis:
        return {
            "matched": [],
            "unmatched_reference": list(reference),
            "unmatched_hypothesis": list(hypothesis),
        }

    model = _get_model()

    # normalize_embeddings=True → vetores unitários → cosine_similarity = produto escalar
    ref_embeddings = model.encode(reference, normalize_embeddings=True, show_progress_bar=False)
    hyp_embeddings = model.encode(hypothesis, normalize_embeddings=True, show_progress_bar=False)

    # Matriz (len(reference), len(hypothesis))
    sim_matrix = cosine_similarity(ref_embeddings, hyp_embeddings)

    matched = []
    matched_hyp_indices: set = set()
    unmatched_ref = []

    for ref_idx, ref_text in enumerate(reference):
        scores = sim_matrix[ref_idx].copy()

        # Mascara índices já casados
        for used_idx in matched_hyp_indices:
            scores[used_idx] = -1.0

        best_hyp_idx = int(np.argmax(scores))
        best_score = float(scores[best_hyp_idx])

        if best_score >= threshold:
            matched.append((ref_text, hypothesis[best_hyp_idx], round(best_score, 4)))
            matched_hyp_indices.add(best_hyp_idx)
        else:
            unmatched_ref.append(ref_text)

    unmatched_hyp = [
        hyp for idx, hyp in enumerate(hypothesis)
        if idx not in matched_hyp_indices
    ]

    return {
        "matched": matched,
        "unmatched_reference": unmatched_ref,
        "unmatched_hypothesis": unmatched_hyp,
    }
