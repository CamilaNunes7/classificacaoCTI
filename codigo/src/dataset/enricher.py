import json
import logging
import re
import time
from pathlib import Path
from typing import List

from google import genai
from google.genai import types

from config.settings import GEMINI_API_KEY, GEMINI_STRUCTURED_MODEL
from src.dataset.schema import TAXONOMY, AREAS, Indicator

log = logging.getLogger(__name__)

_ENRICHMENT_PROMPT = """\
You are an expert in Science, Technology, and Innovation (STI/CTI) indicators.

Classify each indicator in the JSON list below using the taxonomy provided.

## Taxonomy (area → subareas)
{taxonomy}

## Indicators (JSON list of strings)
{indicators_json}

For each indicator return exactly one JSON object with:
- name: the indicator name unchanged
- area: one of the areas above (exact spelling; use "Outros" if truly none fit)
- subarea: the matching subarea (exact spelling; use "Geral" if uncertain)
- keywords: list of 3–5 relevant Portuguese or English keywords
- unit: unit of measure as a string (e.g., "% do PIB", "artigos/ano", "N/A")
- language: "pt" if the name is in Portuguese, "en" if in English

Respond with ONLY a valid JSON array — no markdown, no explanations:
[
  {{"name": "...", "area": "...", "subarea": "...", "keywords": ["..."], "unit": "...", "language": "..."}},
  ...
]
"""


def _taxonomy_str() -> str:
    return "\n".join(f"- {area}: {', '.join(subs)}" for area, subs in TAXONOMY.items())


def _parse_response(text: str, fallback_names: List[str]) -> List[dict]:
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError as e:
        log.error(f"Erro ao parsear resposta de enriquecimento: {e}")
    return [
        {"name": n, "area": "Outros", "subarea": "Geral",
         "keywords": [], "unit": "N/A", "language": "pt"}
        for n in fallback_names
    ]


class DatasetEnricher:
    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY não configurada.")
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def classify_batch(self, names: List[str]) -> List[dict]:
        """Classifica um lote de nomes de indicadores com área/subárea/keywords/unidade."""
        prompt = _ENRICHMENT_PROMPT.format(
            taxonomy=_taxonomy_str(),
            indicators_json=json.dumps(names, ensure_ascii=False),
        )
        for attempt in range(3):
            try:
                response = self.client.models.generate_content(
                    model=GEMINI_STRUCTURED_MODEL,
                    contents=[prompt],
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        max_output_tokens=8192,
                    ),
                )
                classified = _parse_response(response.text, names)
                # Garante que o tamanho bate com o input (matching posicional)
                if len(classified) < len(names):
                    for i in range(len(classified), len(names)):
                        classified.append({
                            "name": names[i], "area": "Outros", "subarea": "Geral",
                            "keywords": [], "unit": "N/A", "language": "pt",
                        })
                return classified[: len(names)]
            except Exception as e:
                if attempt == 2:
                    log.error(f"Enriquecimento falhou após 3 tentativas: {e}")
                    break
                wait = (2 ** attempt) * 5
                log.warning(f"Tentativa {attempt + 1}/3 falhou: {e}. Aguardando {wait}s…")
                time.sleep(wait)
        return [
            {"name": n, "area": "Outros", "subarea": "Geral",
             "keywords": [], "unit": "N/A", "language": "pt"}
            for n in names
        ]

    def enrich_article(
        self,
        article_id: str,
        indicator_names: List[str],
        annotated_by: str = "human",
        batch_size: int = 80,
    ) -> List[Indicator]:
        """Enriquece todos os indicadores de um artigo, processando em lotes."""
        prefix = "h" if annotated_by == "human" else "a"
        enriched: List[Indicator] = []

        for i in range(0, len(indicator_names), batch_size):
            batch = indicator_names[i : i + batch_size]
            log.info(
                f"  [{article_id}] Enriquecendo lote {i // batch_size + 1}"
                f" ({len(batch)} indicadores)…"
            )
            classified = self.classify_batch(batch)

            for j, item in enumerate(classified):
                global_idx = i + j + 1
                area = item.get("area", "Outros")
                if area not in AREAS:
                    area = "Outros"
                enriched.append(
                    Indicator(
                        id=f"{prefix}-{article_id}-{global_idx:03d}",
                        name=item.get("name") or batch[j],
                        area=area,
                        subarea=item.get("subarea", "Geral"),
                        keywords=item.get("keywords", []),
                        source_article=article_id,
                        unit=item.get("unit", "N/A"),
                        excerpt="",
                        language=item.get("language", "pt"),
                        annotated_by=annotated_by,
                    )
                )

        return enriched
