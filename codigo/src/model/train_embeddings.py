"""
Pipeline de treinamento usando embeddings neurais de sentenças.

Modelo de embeddings: paraphrase-multilingual-mpnet-base-v2
Mesmos classificadores, mesmo split (random_state=42) e mesmo
balanceamento (RandomOverSampler) que train.py — garantindo
comparação justa com a abordagem TF-IDF.

Saídas em data/output/ml/results/:
  - metrics_embeddings.json
  - metrics_embeddings_summary.csv
  - emb_confusion_matrix_<model>.png
  - emb_roc_curve.png
"""

import json
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support, accuracy_score,
    roc_curve, auc,
)
from imblearn.over_sampling import RandomOverSampler

warnings.filterwarnings("ignore")

BASE_DIR     = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = BASE_DIR / "data" / "output" / "ml" / "dataset.csv"
RESULTS_DIR  = BASE_DIR / "data" / "output" / "ml" / "results"

RANDOM_STATE    = 42
TEST_SIZE       = 0.2
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"

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


def plot_confusion_matrix(cm, model_name: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Greens", ax=ax,
        xticklabels=["Não-CT&I", "CT&I"],
        yticklabels=["Não-CT&I", "CT&I"],
    )
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de Confusão — {model_name}\n(Embeddings)")
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
    ax.set_title("Curva ROC — Embeddings")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Carregar dataset
    df = pd.read_csv(DATASET_PATH, encoding="utf-8")
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label", "text"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    print(f"Dataset: {len(df)} exemplos")
    print(df["label"].value_counts().rename({1: "CT&I", 0: "Não-CT&I"}).to_string())
    print()

    X = df["text"].values
    y = df["label"].values

    # 2. Mesmo split estratificado de train.py
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Treino: {len(X_train)} | Teste: {len(X_test)}")

    # 3. Gerar embeddings
    print(f"\nCarregando modelo: {EMBEDDING_MODEL}")
    encoder = SentenceTransformer(EMBEDDING_MODEL)

    print("Gerando embeddings do treino...")
    t0 = time.time()
    X_train_emb = encoder.encode(
        X_train.tolist(),
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
    )
    X_test_emb = encoder.encode(
        X_test.tolist(),
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
    )
    embed_time = time.time() - t0
    print(f"Embeddings prontos em {embed_time:.1f}s — dimensão: {X_train_emb.shape[1]}")

    # 4. Balanceamento (mesmo procedimento de train.py)
    ros = RandomOverSampler(random_state=RANDOM_STATE)
    X_train_bal, y_train_bal = ros.fit_resample(X_train_emb, y_train)
    unique, counts = np.unique(y_train_bal, return_counts=True)
    print(f"Treino após oversampling: {dict(zip(['Não-CT&I', 'CT&I'], counts))}\n")

    all_results = {}
    summary_rows = []

    for model_name, clf in MODELS.items():
        print(f"Treinando: {model_name} ...")
        t0 = time.time()
        clf.fit(X_train_bal, y_train_bal)
        train_time = round(time.time() - t0, 2)

        y_pred = clf.predict(X_test_emb)

        print(f"\n=== CLASSIFICATION REPORT — {model_name} ===")
        print(classification_report(y_test, y_pred, target_names=["Não-CTI", "CTI"]))
        print(f"=== CONFUSION MATRIX — {model_name} ===")
        print(confusion_matrix(y_test, y_pred))
        print()

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
            "accuracy":          round(float(acc), 4),
            "precision_cti":     round(float(prec), 4),
            "recall_cti":        round(float(rec), 4),
            "f1_cti":            round(float(f1), 4),
            "precision_nao_cti": round(float(prec_neg), 4),
            "recall_nao_cti":    round(float(rec_neg), 4),
            "f1_nao_cti":        round(float(f1_neg), 4),
            "f1_macro":          round(float(f1_macro), 4),
            "confusion_matrix":  cm.tolist(),
            "classification_report": report,
            "train_time_s":      train_time,
        }

        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X_test_emb)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            result["fpr"]     = fpr.tolist()
            result["tpr"]     = tpr.tolist()
            result["roc_auc"] = round(auc(fpr, tpr), 4)

        all_results[model_name] = result

        safe_name = (
            model_name.lower()
            .replace(" ", "_")
            .replace("ã", "a")
            .replace("ç", "c")
        )
        plot_confusion_matrix(
            cm, model_name,
            RESULTS_DIR / f"emb_confusion_matrix_{safe_name}.png",
        )

        summary_rows.append({
            "Modelo":            model_name,
            "Acurácia":          round(float(acc), 4),
            "Precision CT&I":    round(float(prec), 4),
            "Recall CT&I":       round(float(rec), 4),
            "F1 CT&I":           round(float(f1), 4),
            "F1 Não-CT&I":       round(float(f1_neg), 4),
            "F1 Macro":          round(float(f1_macro), 4),
            "ROC AUC":           round(result.get("roc_auc", 0), 4),
            "Tempo treino (s)":  train_time,
        })

        print(
            f"  Acurácia: {acc:.4f} | F1 CT&I: {f1:.4f} | "
            f"F1 Macro: {f1_macro:.4f} | AUC: {result.get('roc_auc', '-')} | "
            f"Tempo: {train_time}s"
        )

    # Salvar JSON
    with open(RESULTS_DIR / "metrics_embeddings.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # Salvar CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        RESULTS_DIR / "metrics_embeddings_summary.csv",
        index=False, encoding="utf-8",
    )

    # ROC
    plot_roc(all_results, RESULTS_DIR / "emb_roc_curve.png")

    print()
    print("=== RESULTADO FINAL (Embeddings) ===")
    print(summary_df.to_string(index=False))
    print(f"\nArquivos salvos em: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
