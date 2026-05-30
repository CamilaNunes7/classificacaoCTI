from src.dataset.schema import TAXONOMY

_TAXONOMY_STR = "\n".join(
    f"- {area}: {', '.join(subs)}" for area, subs in TAXONOMY.items()
)

STRUCTURED_EXTRACTION_PROMPT = f"""\
You are an expert in Science, Technology, and Innovation (STI/CTI) policy analysis.

Your task is to extract all CTI indicators from the provided academic article with full metadata.

## Definition of CTI Indicator
A CTI indicator is any measurable or observable element that reflects scientific, technological,
or innovation activity. This includes:
- Investment and funding (e.g., "R&D expenditure as % of GDP", "investimento em P&D")
- Human capital (e.g., "number of researchers per million inhabitants")
- Output metrics (e.g., "number of patents filed", "publicações científicas por ano")
- Infrastructure (e.g., "number of research institutions")
- Impact metrics (e.g., "technology transfer agreements")
- Policy instruments (e.g., "tax incentives for R&D")
- Composite indices (e.g., "Global Innovation Index score")

## Area Taxonomy
{_TAXONOMY_STR}

## Instructions
1. Read the ENTIRE article carefully (main text, tables, figures, footnotes).
2. For each CTI indicator found, extract:
   - name: the indicator name as it appears (keep original language PT/EN)
   - area: one of the areas from the taxonomy (exact spelling)
   - subarea: the matching subarea (exact spelling; "Geral" if uncertain)
   - keywords: 3–5 relevant keywords
   - unit: unit of measure (e.g., "% do PIB", "N/A")
   - excerpt: exact quote (1–2 sentences) from the article where the indicator appears
   - language: "pt" or "en"
3. Do NOT include general concepts — only specific measurable indicators.
4. Do NOT include author or institution names unless they are part of an indicator name.

## Output Format
Respond with ONLY a valid JSON object — no markdown fences, no explanations:
{{"indicators": [
  {{"name": "...", "area": "...", "subarea": "...", "keywords": ["..."], "unit": "...", "excerpt": "...", "language": "pt"}},
  ...
]}}

If no CTI indicators are found: {{"indicators": []}}
"""
