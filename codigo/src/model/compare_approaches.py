"""
Gera tabela e gráficos comparando TF-IDF vs Embeddings.

Lê:
  data/output/ml/results/metrics.json              (TF-IDF)
  data/output/ml/results/metrics_embeddings.json   (Embeddings)

Saídas:
  data/output/ml/results/comparison_tfidf_vs_embeddings.csv
  data/output/ml/results/comparison_tfidf_vs_embeddings.png
  data/output/ml/results/comparison_roc_combined.png
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path

BASE_DIR    = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = BASE_DIR / "data" / "output" / "ml" / "results"

COLORS = {"TF-IDF": "#4C72B0", "Embeddings": "#DD8452"}
METRICS = ["Acurácia", "Precision CT&I", "Recall CT&I", "F1 CT&I", "F1 Macro", "ROC AUC"]


def load_summary(path: Path, label: str) -> pd.DataFrame:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for model_name, r in data.items():
        rows.append({
            "Modelo":         model_name,
            "Vetorização":    label,
            "Acurácia":       r["accuracy"],
            "Precision CT&I": r["precision_cti"],
            "Recall CT&I":    r["recall_cti"],
            "F1 CT&I":        r["f1_cti"],
            "F1 Não-CT&I":    r["f1_nao_cti"],
            "F1 Macro":       r["f1_macro"],
            "ROC AUC":        r.get("roc_auc", 0.0),
        })
    return pd.DataFrame(rows)


def plot_comparison_bars(df: pd.DataFrame, out_path: Path):
    models = df["Modelo"].unique()
    x = np.arange(len(models))
    width = 0.35

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, metric in enumerate(METRICS):
        ax = axes[i]
        for j, vet in enumerate(["TF-IDF", "Embeddings"]):
            subset = df[df["Vetorização"] == vet].set_index("Modelo").reindex(models)
            vals = subset[metric].values
            bars = ax.bar(
                x + (j - 0.5) * width, vals, width,
                label=vet, color=COLORS[vet], alpha=0.85, edgecolor="white"
            )
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7
                )

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=13, ha="right", fontsize=8)
        ymin = max(0.0, df[metric].min() - 0.05)
        ax.set_ylim(ymin, 1.02)
        ax.set_title(metric, fontsize=10)
        ax.set_ylabel("Score", fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.suptitle(
        "TF-IDF vs Embeddings Neurais — Comparação no Conjunto de Teste",
        fontsize=12, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_combined(tfidf_path: Path, emb_path: Path, out_path: Path):
    tfidf_data = json.loads(tfidf_path.read_text(encoding="utf-8"))
    emb_data   = json.loads(emb_path.read_text(encoding="utf-8"))

    models = list(tfidf_data.keys())
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), sharey=True)

    if len(models) == 1:
        axes = [axes]

    linestyles = {"TF-IDF": "-", "Embeddings": "--"}
    sources = {"TF-IDF": tfidf_data, "Embeddings": emb_data}

    for ax, model_name in zip(axes, models):
        for vet, data in sources.items():
            r = data.get(model_name, {})
            if "fpr" in r:
                ax.plot(
                    r["fpr"], r["tpr"],
                    linestyle=linestyles[vet],
                    color=COLORS[vet],
                    label=f"{vet} (AUC={r['roc_auc']:.3f})"
                )
        ax.plot([0, 1], [0, 1], "k:", alpha=0.4)
        ax.set_title(model_name, fontsize=9)
        ax.set_xlabel("FPR")
        if ax == axes[0]:
            ax.set_ylabel("TPR")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(alpha=0.3)

    plt.suptitle("Curvas ROC — TF-IDF vs Embeddings", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    tfidf_path = RESULTS_DIR / "metrics.json"
    emb_path   = RESULTS_DIR / "metrics_embeddings.json"

    if not tfidf_path.exists():
        print(f"Arquivo não encontrado: {tfidf_path}\nExecute src/model/train.py primeiro.")
        return
    if not emb_path.exists():
        print(f"Arquivo não encontrado: {emb_path}\nExecute src/model/train_embeddings.py primeiro.")
        return

    df_tfidf = load_summary(tfidf_path, "TF-IDF")
    df_emb   = load_summary(emb_path,   "Embeddings")
    df_all   = pd.concat([df_tfidf, df_emb], ignore_index=True)
    df_all   = df_all.sort_values(["Modelo", "Vetorização"]).reset_index(drop=True)

    # Tabela comparativa
    out_csv = RESULTS_DIR / "comparison_tfidf_vs_embeddings.csv"
    df_all.to_csv(out_csv, index=False, encoding="utf-8")

    print("=== COMPARAÇÃO TF-IDF vs EMBEDDINGS ===\n")
    pivot = df_all.pivot_table(
        index="Modelo", columns="Vetorização", values=METRICS
    )
    print(pivot.round(4).to_string())

    # Delta (Embeddings - TF-IDF) para cada métrica
    print("\n=== DELTA (Embeddings − TF-IDF) ===")
    for metric in METRICS:
        sub = df_all.pivot(index="Modelo", columns="Vetorização", values=metric)
        sub["Delta"] = sub.get("Embeddings", 0) - sub.get("TF-IDF", 0)
        print(f"\n{metric}:")
        print(sub[["TF-IDF", "Embeddings", "Delta"]].round(4).to_string())

    # Gráficos
    out_bar = RESULTS_DIR / "comparison_tfidf_vs_embeddings.png"
    plot_comparison_bars(df_all, out_bar)

    out_roc = RESULTS_DIR / "comparison_roc_combined.png"
    plot_roc_combined(tfidf_path, emb_path, out_roc)

    print(f"\nCSV  → {out_csv}")
    print(f"PNG  → {out_bar}")
    print(f"ROC  → {out_roc}")


if __name__ == "__main__":
    main()
