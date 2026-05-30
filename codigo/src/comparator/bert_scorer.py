import logging
from typing import Any, Dict, List

import numpy as np
import torch

from config.settings import BERTSCORE_MODEL, BERTSCORE_THRESHOLD

log = logging.getLogger(__name__)

# Singletons: carregados uma vez e reutilizados
_model = None
_tokenizer = None


def _get_model_tokenizer():
    global _model, _tokenizer
    if _model is None:
        from transformers import AutoModel, AutoTokenizer

        log.info(f"Carregando modelo BERTScore: {BERTSCORE_MODEL}")
        _tokenizer = AutoTokenizer.from_pretrained(BERTSCORE_MODEL)
        _model = AutoModel.from_pretrained(BERTSCORE_MODEL)
        _model.eval()
    return _model, _tokenizer


def _encode_sentences(sentences: List[str], batch_size: int = 32) -> List[torch.Tensor]:
    """
    Retorna lista de tensores de embeddings de tokens para cada sentença.
    Cada tensor tem shape (n_tokens_conteudo, hidden_dim) com vetores L2-normalizados.
    Tokens especiais ([CLS]/[SEP] ou <s>/</s>) são removidos.
    """
    model, tokenizer = _get_model_tokenizer()
    device = next(model.parameters()).device
    all_embeddings: List[torch.Tensor] = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            output = model(**encoded)

        # last_hidden_state: (B, L, D)
        token_embs = output.last_hidden_state
        attn_mask = encoded["attention_mask"]

        # L2-normaliza cada embedding de token
        norms = token_embs.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        token_embs = token_embs / norms

        for emb, mask in zip(token_embs, attn_mask):
            real_len = int(mask.sum().item())
            # Remove tokens especiais: posição 0 ([CLS]/<s>) e última posição real ([SEP]/</s>)
            content_len = real_len - 2
            if content_len <= 0:
                # Fallback: usa pelo menos o token [CLS] para textos extremamente curtos
                content_emb = emb[0:1].cpu()
            else:
                content_emb = emb[1 : 1 + content_len].cpu()
            all_embeddings.append(content_emb)

    return all_embeddings


def _bertscore_f1(ref_emb: torch.Tensor, hyp_emb: torch.Tensor) -> float:
    """
    Calcula BERTScore F1 entre dois conjuntos de embeddings de tokens.

    Recall  = média das máximas similaridades de cada token da REF com a HYP.
    Precision = média das máximas similaridades de cada token da HYP com a REF.
    F1 = média harmônica de Precision e Recall.
    """
    # sim[i, j] = cos_sim(ref_token_i, hyp_token_j)  — vetores já normalizados
    sim = torch.mm(ref_emb, hyp_emb.T)  # (L_ref, L_hyp)

    recall = float(sim.max(dim=1)[0].mean().item())
    precision = float(sim.max(dim=0)[0].mean().item())

    denom = precision + recall
    if denom < 1e-10:
        return 0.0
    return 2.0 * precision * recall / denom


def match(
    reference: List[str],
    hypothesis: List[str],
    threshold: float = BERTSCORE_THRESHOLD,
) -> Dict[str, Any]:
    """
    Matching via BERTScore F1 (similaridade contextual token a token).

    Usa texto bruto (sem normalização de acentos/lowercase) para preservar
    a qualidade dos embeddings contextuais.

    Estratégia:
    - Codifica todos os indicadores (referência + hipótese) em uma única passagem.
    - Constrói matriz de BERTScore F1 (|ref| × |hyp|).
    - Matching guloso: para cada item da referência, encontra o melhor item
      ainda não casado na hipótese com BERTScore F1 ≥ threshold.

    Retorna:
      matched: lista de (ref_text, hyp_text, score) para pares acima do threshold
      unmatched_reference: falsos negativos
      unmatched_hypothesis: falsos positivos
    """
    if not reference or not hypothesis:
        return {
            "matched": [],
            "unmatched_reference": list(reference),
            "unmatched_hypothesis": list(hypothesis),
        }

    log.info(f"  BERTScore: codificando {len(reference) + len(hypothesis)} sentenças...")

    # Codifica todas as sentenças de uma vez (cada uma apenas uma vez)
    all_embs = _encode_sentences(reference + hypothesis)
    ref_embs = all_embs[: len(reference)]
    hyp_embs = all_embs[len(reference) :]

    # Monta matriz de scores (|ref| × |hyp|)
    n_ref = len(reference)
    n_hyp = len(hypothesis)
    score_matrix = np.zeros((n_ref, n_hyp), dtype=np.float32)

    for i in range(n_ref):
        for j in range(n_hyp):
            score_matrix[i, j] = _bertscore_f1(ref_embs[i], hyp_embs[j])

    # Matching guloso (mesmo esquema do semantic_matcher)
    matched: List = []
    matched_hyp_indices: set = set()
    unmatched_ref: List[str] = []

    for ref_idx, ref_text in enumerate(reference):
        scores = score_matrix[ref_idx].copy()
        for used_idx in matched_hyp_indices:
            scores[used_idx] = -1.0

        best_hyp_idx = int(np.argmax(scores))
        best_score = float(scores[best_hyp_idx])

        if best_score >= threshold:
            matched.append(
                (ref_text, hypothesis[best_hyp_idx], round(best_score, 4))
            )
            matched_hyp_indices.add(best_hyp_idx)
        else:
            unmatched_ref.append(ref_text)

    unmatched_hyp = [
        hyp
        for idx, hyp in enumerate(hypothesis)
        if idx not in matched_hyp_indices
    ]

    return {
        "matched": matched,
        "unmatched_reference": unmatched_ref,
        "unmatched_hypothesis": unmatched_hyp,
    }
