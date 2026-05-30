EXTRACTION_PROMPT = """\
You are an expert in Science, Technology, and Innovation (STI/CTI — Ciência, Tecnologia e Inovação) policy analysis.

Your task is to extract all CTI indicators from the provided academic article.

## Definition of CTI Indicator
A CTI indicator is any measurable or observable element that reflects scientific, technological, or innovation activity. This includes:

- Investment and funding metrics (e.g., "R&D expenditure as % of GDP", "investimento em P&D")
- Human capital metrics (e.g., "number of researchers per million inhabitants", "pesquisadores por habitante")
- Output metrics (e.g., "number of patents filed", "publicações científicas por ano")
- Infrastructure metrics (e.g., "number of research institutions", "laboratórios credenciados")
- Impact metrics (e.g., "technology transfer agreements", "spin-offs gerados")
- Policy instruments (e.g., "tax incentives for R&D", "bolsas de pós-graduação")
- Composite indices (e.g., "Global Innovation Index score", "Índice de Inovação")
- Any other quantitative or qualitative measure used to assess STI performance

## Instructions
1. Read the ENTIRE article carefully.
2. Extract ALL CTI indicators mentioned, whether in the main text, tables, figures, or footnotes.
3. Preserve the indicator name as it appears in the article (keep the original language: Portuguese or English).
4. If an indicator has both a Portuguese and an English name, extract the primary one used in the article.
5. Do NOT include general concepts or topics — only specific indicators.
6. Do NOT include author names, institution names, or geographic locations unless they are part of an indicator name.

## Output Format
Respond with ONLY a valid JSON object in this exact format — no markdown fences, no explanations:

{"indicators": ["indicator name 1", "indicator name 2", "indicator name 3"]}

If no CTI indicators are found, respond with: {"indicators": []}
"""
