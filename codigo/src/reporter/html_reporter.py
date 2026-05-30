import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Environment, BaseLoader

log = logging.getLogger(__name__)

METHODS = ["exact", "fuzzy", "semantic", "bertscore"]
PAIRS = ["human_vs_gemini", "human_vs_claude", "gemini_vs_claude"]

PAIR_LABELS = {
    "human_vs_gemini": "Humano × Gemini",
    "human_vs_claude": "Humano × Claude",
    "gemini_vs_claude": "Gemini × Claude",
}

METHOD_LABELS = {
    "exact": "Exato",
    "fuzzy": "Fuzzy",
    "semantic": "Semântico",
    "bertscore": "BERTScore",
}

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <title>Relatório CTI — Avaliação de Extração de Indicadores</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 40px; color: #222; }
    h1 { color: #1a237e; }
    h2 { color: #283593; border-bottom: 2px solid #3f51b5; padding-bottom: 4px; }
    h3 { color: #3949ab; }
    table { border-collapse: collapse; width: 100%; margin-bottom: 24px; font-size: 13px; }
    th { background: #3f51b5; color: white; padding: 8px 12px; text-align: left; }
    td { padding: 6px 12px; border-bottom: 1px solid #ddd; }
    tr:nth-child(even) { background: #f5f5f5; }
    .agg { background: #e8eaf6 !important; font-weight: bold; }
    .high { background: #c8e6c9 !important; }
    .mid  { background: #fff9c4 !important; }
    .low  { background: #ffcdd2 !important; }
    .section { margin-bottom: 48px; }
    .meta { color: #666; font-size: 13px; margin-bottom: 32px; }
    details { margin: 8px 0; }
    summary { cursor: pointer; font-weight: bold; color: #3f51b5; }
    .match-list { font-size: 12px; font-family: monospace; margin: 8px 0 8px 24px; }
    .match-item { margin: 2px 0; }
    .ref  { color: #1b5e20; }
    .hyp  { color: #0d47a1; }
    .miss { color: #b71c1c; }
    .extra { color: #e65100; }
  </style>
</head>
<body>
  <h1>Relatório de Avaliação — Extração de Indicadores CTI</h1>
  <div class="meta">
    Gerado em: {{ generated_at }}<br>
    Artigos avaliados: {{ n_articles }}<br>
    Pares comparados: Humano×Gemini, Humano×Claude, Gemini×Claude<br>
    Métodos: Exato, Fuzzy (WRatio ≥ 80), Semântico (cosseno ≥ 0,80), BERTScore F1 (≥ 0,85)
  </div>

  {% if area_metrics %}
  <div class="section">
    <h2>Métricas por Área Temática (Fuzzy — Humano × Gemini)</h2>
    <p class="meta">Dataset enriquecido com Gemini 2.5 Pro — classificação automática de área via taxonomia CT&I.</p>
    <table>
      <tr>
        <th>Área</th>
        <th>Ref. Humano (#)</th>
        <th>Extraído IA (#)</th>
        <th>Precisão</th>
        <th>Recall</th>
        <th>F1</th>
      </tr>
      {% for row in area_metrics %}
      <tr>
        <td><strong>{{ row.area }}</strong></td>
        <td>{{ row.ref_count }}</td>
        <td>{{ row.hyp_count }}</td>
        <td class="{{ 'high' if row.precision >= 0.8 else ('mid' if row.precision >= 0.5 else 'low') }}">{{ "%.4f"|format(row.precision) }}</td>
        <td class="{{ 'high' if row.recall >= 0.8 else ('mid' if row.recall >= 0.5 else 'low') }}">{{ "%.4f"|format(row.recall) }}</td>
        <td class="{{ 'high' if row.f1 >= 0.8 else ('mid' if row.f1 >= 0.5 else 'low') }}">{{ "%.4f"|format(row.f1) }}</td>
      </tr>
      {% endfor %}
    </table>
  </div>
  {% endif %}

  <div class="section">
    <h2>Sumário Geral (Macro-Média)</h2>
    <table>
      <tr>
        <th>Par</th>
        <th>Método</th>
        <th>Precisão</th>
        <th>Recall</th>
        <th>F1</th>
      </tr>
      {% for pair in pairs %}
        {% for method in methods %}
          {% set agg = aggregates[pair][method] %}
          <tr class="agg">
            <td>{{ pair_labels[pair] }}</td>
            <td>{{ method_labels[method] }}</td>
            <td>{{ "%.4f"|format(agg.precision) }}</td>
            <td>{{ "%.4f"|format(agg.recall) }}</td>
            <td>{{ "%.4f"|format(agg.f1) }}</td>
          </tr>
        {% endfor %}
      {% endfor %}
    </table>
  </div>

  {% for pair in pairs %}
  <div class="section">
    <h2>{{ pair_labels[pair] }}</h2>
    <table>
      <tr>
        <th>Artigo</th>
        <th>Ref. (#)</th>
        <th>Hyp. (#)</th>
        {% for method in methods %}
          <th>{{ method_labels[method] }} P</th>
          <th>{{ method_labels[method] }} R</th>
          <th>{{ method_labels[method] }} F1</th>
        {% endfor %}
      </tr>
      {% for result in results %}
        {% if pair in result.pairs %}
          {% set pair_data = result.pairs[pair] %}
          {% set ref_name = pair.split('_vs_')[0] %}
          {% set hyp_name = pair.split('_vs_')[1] %}
          <tr>
            <td>{{ result.article_id }}</td>
            <td>{{ result.indicator_counts.get(ref_name, 0) }}</td>
            <td>{{ result.indicator_counts.get(hyp_name, 0) }}</td>
            {% for method in methods %}
              {% set m = pair_data[method] %}
              <td class="{{ 'high' if m.precision >= 0.8 else ('mid' if m.precision >= 0.5 else 'low') }}">{{ "%.4f"|format(m.precision) }}</td>
              <td class="{{ 'high' if m.recall >= 0.8 else ('mid' if m.recall >= 0.5 else 'low') }}">{{ "%.4f"|format(m.recall) }}</td>
              <td class="{{ 'high' if m.f1 >= 0.8 else ('mid' if m.f1 >= 0.5 else 'low') }}">{{ "%.4f"|format(m.f1) }}</td>
            {% endfor %}
          </tr>
        {% endif %}
      {% endfor %}
    </table>

    <h3>Análise Qualitativa por Artigo</h3>
    {% for result in results %}
      {% if pair in result.pairs %}
        {% set pair_data = result.pairs[pair] %}
        <details>
          <summary>{{ result.article_id }} — Semântico</summary>
          <div class="match-list">
            <strong>Casamentos (TP):</strong><br>
            {% for ref_t, hyp_t, score in pair_data.semantic.matched %}
              <div class="match-item">
                <span class="ref">REF: {{ ref_t }}</span> →
                <span class="hyp">HYP: {{ hyp_t }}</span>
                (score: {{ "%.3f"|format(score) }})
              </div>
            {% else %}
              <div class="match-item">Nenhum.</div>
            {% endfor %}
            <br>
            <strong>Ausentes na hipótese (FN):</strong><br>
            {% for item in pair_data.semantic.unmatched_reference %}
              <div class="match-item miss">✗ {{ item }}</div>
            {% else %}
              <div class="match-item">Nenhum.</div>
            {% endfor %}
            <br>
            <strong>Extras na hipótese (FP):</strong><br>
            {% for item in pair_data.semantic.unmatched_hypothesis %}
              <div class="match-item extra">+ {{ item }}</div>
            {% else %}
              <div class="match-item">Nenhum.</div>
            {% endfor %}
          </div>
        </details>
        <details>
          <summary>{{ result.article_id }} — BERTScore</summary>
          <div class="match-list">
            <strong>Casamentos (TP):</strong><br>
            {% for ref_t, hyp_t, score in pair_data.bertscore.matched %}
              <div class="match-item">
                <span class="ref">REF: {{ ref_t }}</span> →
                <span class="hyp">HYP: {{ hyp_t }}</span>
                (score: {{ "%.3f"|format(score) }})
              </div>
            {% else %}
              <div class="match-item">Nenhum.</div>
            {% endfor %}
            <br>
            <strong>Ausentes na hipótese (FN):</strong><br>
            {% for item in pair_data.bertscore.unmatched_reference %}
              <div class="match-item miss">✗ {{ item }}</div>
            {% else %}
              <div class="match-item">Nenhum.</div>
            {% endfor %}
            <br>
            <strong>Extras na hipótese (FP):</strong><br>
            {% for item in pair_data.bertscore.unmatched_hypothesis %}
              <div class="match-item extra">+ {{ item }}</div>
            {% else %}
              <div class="match-item">Nenhum.</div>
            {% endfor %}
          </div>
        </details>
      {% endif %}
    {% endfor %}
  </div>
  {% endfor %}

</body>
</html>
"""


def _compute_aggregates(all_results: List[Dict[str, Any]]) -> Dict:
    """Calcula macro-médias de P/R/F1 por par e método."""
    sums: Dict[str, Dict[str, Dict[str, float]]] = {}
    counts: Dict[str, Dict[str, int]] = {}

    for result in all_results:
        for pair_key, pair_data in result["pairs"].items():
            if pair_key not in sums:
                sums[pair_key] = {}
                counts[pair_key] = {}
            for method in METHODS:
                if method not in sums[pair_key]:
                    sums[pair_key][method] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
                    counts[pair_key][method] = 0
                m = pair_data.get(method, {})
                sums[pair_key][method]["precision"] += m.get("precision", 0.0)
                sums[pair_key][method]["recall"] += m.get("recall", 0.0)
                sums[pair_key][method]["f1"] += m.get("f1", 0.0)
                counts[pair_key][method] += 1

    aggregates: Dict = {}
    for pair_key, methods in sums.items():
        aggregates[pair_key] = {}
        for method, totals in methods.items():
            n = counts[pair_key][method] or 1
            aggregates[pair_key][method] = {
                "precision": round(totals["precision"] / n, 4),
                "recall": round(totals["recall"] / n, 4),
                "f1": round(totals["f1"] / n, 4),
            }
    return aggregates


def generate_html(
    all_results: List[Dict[str, Any]],
    output_path: Path,
    area_metrics: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Gera o relatório HTML completo. Inclui seção por área se area_metrics for fornecido."""
    env = Environment(loader=BaseLoader())
    template = env.from_string(HTML_TEMPLATE)

    aggregates = _compute_aggregates(all_results)
    active_pairs = [p for p in PAIRS if p in aggregates]

    rendered = template.render(
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        n_articles=len(all_results),
        results=all_results,
        pairs=active_pairs,
        methods=METHODS,
        pair_labels=PAIR_LABELS,
        method_labels=METHOD_LABELS,
        aggregates=aggregates,
        area_metrics=area_metrics,
    )

    output_path.write_text(rendered, encoding="utf-8")
    log.info(f"Relatório HTML salvo em: {output_path}")
