"""
Análise de erros do melhor modelo (SVM Linear).
Identifica padrões nos falsos positivos e falsos negativos.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import RandomOverSampler

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = BASE_DIR / "data" / "output" / "ml" / "dataset.csv"
RESULTS_DIR = BASE_DIR / "data" / "output" / "ml" / "results"

RANDOM_STATE = 42
TFIDF_PARAMS = dict(ngram_range=(1, 2), max_features=10_000, sublinear_tf=True, min_df=2)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATASET_PATH, encoding="utf-8")
    X, y = df["text"].values, df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    df_test = df.iloc[
        pd.Series(range(len(df))).sample(frac=1, random_state=RANDOM_STATE).values
    ].reset_index(drop=True)

    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    ros = RandomOverSampler(random_state=RANDOM_STATE)
    X_train_bal, y_train_bal = ros.fit_resample(X_train_tfidf, y_train)

    model = CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=RANDOM_STATE))
    model.fit(X_train_bal, y_train_bal)
    y_pred = model.predict(X_test_tfidf)

    # Montar dataframe de erros
    error_df = pd.DataFrame({
        "text": X_test,
        "real": y_test,
        "predito": y_pred,
    })
    error_df["erro"] = error_df["real"] != error_df["predito"]

    fp = error_df[(error_df["real"] == 0) & (error_df["predito"] == 1)]  # Falso Positivo
    fn = error_df[(error_df["real"] == 1) & (error_df["predito"] == 0)]  # Falso Negativo

    # Salvar exemplos de erros
    errors_path = RESULTS_DIR / "error_analysis.csv"
    error_df[error_df["erro"]].assign(
        tipo=lambda d: d.apply(
            lambda r: "Falso Positivo (não-CT&I classificado como CT&I)"
                      if r["predito"] == 1 else "Falso Negativo (CT&I perdido)", axis=1
        )
    ).to_csv(errors_path, index=False, encoding="utf-8")

    print(f"Total de erros: {error_df['erro'].sum()} / {len(error_df)}")
    print(f"  Falsos Positivos (não-CT&I → CT&I): {len(fp)}")
    print(f"  Falsos Negativos (CT&I → não-CT&I): {len(fn)}")
    print()

    print("=== EXEMPLOS DE FALSOS POSITIVOS (não-CT&I classificado como CT&I) ===")
    for _, row in fp.head(10).iterrows():
        print(f"  • {row['text']}")

    print()
    print("=== EXEMPLOS DE FALSOS NEGATIVOS (CT&I não encontrado) ===")
    for _, row in fn.head(10).iterrows():
        print(f"  • {row['text']}")

    # Gráfico distribuição de erros por tamanho de texto
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    error_df["n_tokens"] = error_df["text"].str.split().str.len()

    for ax, (subset, title) in zip(axes, [(fp, "Falsos Positivos"), (fn, "Falsos Negativos")]):
        subset_tokens = error_df.loc[subset.index, "n_tokens"]
        ax.hist(subset_tokens, bins=15, color="tomato", edgecolor="white")
        ax.set_title(f"{title} (n={len(subset)})")
        ax.set_xlabel("Número de palavras")
        ax.set_ylabel("Frequência")

    plt.suptitle("Distribuição do comprimento dos textos com erro — SVM Linear")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "error_distribution.png", dpi=150)
    plt.close(fig)

    print(f"\nArquivos salvos em: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
