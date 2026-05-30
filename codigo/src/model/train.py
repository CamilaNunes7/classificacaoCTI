"""
Pipeline de treinamento e avaliação do classificador CT&I.

Modelos treinados:
  - Regressão Logística (TF-IDF)
  - SVM linear (TF-IDF)
  - Random Forest (TF-IDF)

Balanceamento: RandomOverSampler na classe CT&I (treino apenas).
Avaliação: conjunto de teste desbalanceado (proporção real).

Saídas em data/output/ml/results/:
  - metrics.json         — todas as métricas
  - metrics_summary.csv  — tabela comparativa
  - confusion_matrix_<model>.png
  - roc_curve.png
"""

import json
import warnings
import pandas as pd
import numpy as np
# pyrefly: ignore [missing-import]
import matplotlib
matplotlib.use("Agg")
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support, accuracy_score,
    roc_curve, auc
)
from sklearn.calibration import CalibratedClassifierCV
# pyrefly: ignore [missing-import]
from imblearn.over_sampling import RandomOverSampler

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = BASE_DIR / "data" / "output" / "ml" / "dataset.csv"
RESULTS_DIR = BASE_DIR / "data" / "output" / "ml" / "results"


# ── Configurações ────────────────────────────────────────────────────────────

RANDOM_STATE = 42
TEST_SIZE = 0.2

TFIDF_PARAMS = dict(
    ngram_range=(1, 2),
    max_features=10_000,
    sublinear_tf=True,
    min_df=2,
)

MODELS = {
    "Regressão Logística": LogisticRegression(
        max_iter=1000, class_weight=None, random_state=RANDOM_STATE
    ),
    "SVM Linear": CalibratedClassifierCV(
        LinearSVC(max_iter=2000, random_state=RANDOM_STATE)
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
    ),
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm, model_name: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Não-CT&I", "CT&I"],
        yticklabels=["Não-CT&I", "CT&I"],
    )
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de Confusão — {model_name}")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc(results: dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, r in results.items():
        if "fpr" in r:
            ax.plot(r["fpr"], r["tpr"], label=f"{name} (AUC={r['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Aleatório")
    ax.set_xlabel("Taxa de Falsos Positivos")
    ax.set_ylabel("Taxa de Verdadeiros Positivos")
    ax.set_title("Curva ROC — Comparação de Modelos")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metrics_bar(summary_df: pd.DataFrame, out_path: Path):
    metrics = ["F1 CT&I", "F1 Macro", "Precision CT&I", "Recall CT&I", "Acurácia"]
    available = [m for m in metrics if m in summary_df.columns]
    df_plot = summary_df.set_index("Modelo")[available]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(df_plot))
    width = 0.15
    for i, col in enumerate(available):
        ax.bar(x + i * width, df_plot[col], width, label=col)
    ax.set_xticks(x + width * (len(available) - 1) / 2)
    ax.set_xticklabels(df_plot.index, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Comparação de Modelos — Conjunto de Teste (desbalanceado)")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Pipeline principal ───────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Carregar dataset
    df = pd.read_csv(DATASET_PATH, encoding="utf-8")
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label", "text"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    print(f"Dataset carregado: {len(df)} exemplos")
    print(df["label"].value_counts().rename({1: "CT&I", 0: "Não-CT&I"}).to_string())
    print()

    X = df["text"].values
    y = df["label"].values

    # 2. Split estratificado (mantém proporção real no teste)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Treino: {len(X_train)} | Teste: {len(X_test)}")
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"Treino antes do balanceamento: {dict(zip(['Não-CT&I','CT&I'], counts))}")

    # 3. Vetorização TF-IDF
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 4. Balanceamento do treino (oversampling da classe CT&I)
    ros = RandomOverSampler(random_state=RANDOM_STATE)
    X_train_bal, y_train_bal = ros.fit_resample(X_train_tfidf, y_train)
    unique, counts = np.unique(y_train_bal, return_counts=True)
    print(f"Treino após oversampling:      {dict(zip(['Não-CT&I','CT&I'], counts))}")
    print()

    all_results = {}
    summary_rows = []

    for model_name, model in MODELS.items():
        print(f"Treinando: {model_name} ...")
        model.fit(X_train_bal, y_train_bal)
        y_pred = model.predict(X_test_tfidf)

        # Métricas
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, labels=[1], average="binary"
        )
        prec_neg, rec_neg, f1_neg, _ = precision_recall_fscore_support(
            y_test, y_pred, labels=[0], average="binary"
        )
        _, _, f1_macro, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro"
        )
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(
            y_test, y_pred,
            target_names=["Não-CT&I", "CT&I"],
            output_dict=True,
        )

        result = {
            "accuracy": round(float(acc), 4),
            "precision_cti": round(float(prec), 4),
            "recall_cti": round(float(rec), 4),
            "f1_cti": round(float(f1), 4),
            "precision_nao_cti": round(float(prec_neg), 4),
            "recall_nao_cti": round(float(rec_neg), 4),
            "f1_nao_cti": round(float(f1_neg), 4),
            "f1_macro": round(float(f1_macro), 4),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        }

        # Curva ROC (requer probabilidades)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_tfidf)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            result["fpr"] = fpr.tolist()
            result["tpr"] = tpr.tolist()
            result["roc_auc"] = round(roc_auc, 4)

        all_results[model_name] = result

        # Matriz de confusão
        plot_confusion_matrix(
            cm, model_name,
            RESULTS_DIR / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
        )

        summary_rows.append({
            "Modelo": model_name,
            "Acurácia": round(float(acc), 4),
            "Precision CT&I": round(float(prec), 4),
            "Recall CT&I": round(float(rec), 4),
            "F1 CT&I": round(float(f1), 4),
            "F1 Não-CT&I": round(float(f1_neg), 4),
            "F1 Macro": round(float(f1_macro), 4),
            "ROC AUC": round(result.get("roc_auc", 0), 4),
        })

        print(f"  Acurácia: {acc:.4f} | F1 CT&I: {f1:.4f} | F1 Macro: {f1_macro:.4f} | AUC: {result.get('roc_auc', '-')}")

    # Salvar métricas JSON
    metrics_path = RESULTS_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # Salvar tabela CSV
    summary_df = pd.DataFrame(summary_rows)
    csv_path = RESULTS_DIR / "metrics_summary.csv"
    summary_df.to_csv(csv_path, index=False, encoding="utf-8")

    # Gráficos
    plot_roc(all_results, RESULTS_DIR / "roc_curve.png")
    plot_metrics_bar(summary_df, RESULTS_DIR / "metrics_bar.png")

    print()
    print("=== RESULTADO FINAL ===")
    print(summary_df.to_string(index=False))
    print()
    print(f"Arquivos salvos em: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
