"""
Explicabilidade local com LIME para o pipeline
paraphrase-multilingual-mpnet-base-v2 + SVM Linear.

Seleciona:
  - 3 Falsos Positivos (não-CT&I predito como CT&I)
  - 3 Falsos Negativos (CT&I predito como não-CT&I)
  - 3 acertos da classe CT&I
  - 3 acertos da classe não-CT&I

Para cada exemplo imprime os 5 tokens com maior peso LIME e salva
os resultados em data/output/ml/results/lime_explanations.csv
"""

import warnings
import numpy as np
import pandas as pd

from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import RandomOverSampler
from lime.lime_text import LimeTextExplainer

warnings.filterwarnings("ignore")

BASE_DIR     = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = BASE_DIR / "data" / "output" / "ml" / "dataset.csv"
RESULTS_DIR  = BASE_DIR / "data" / "output" / "ml" / "results"

RANDOM_STATE    = 42
TEST_SIZE       = 0.2
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"
NUM_FEATURES    = 5
NUM_SAMPLES     = 3       # exemplos por categoria
NUM_LIME_SAMPLES = 1000   # perturbações internas do LIME


# ---------------------------------------------------------------------------
# 1. Dataset e split
# ---------------------------------------------------------------------------
print("Carregando dataset...")
df = pd.read_csv(DATASET_PATH, encoding="utf-8")
df["label"] = pd.to_numeric(df["label"], errors="coerce")
df = df.dropna(subset=["label", "text"]).reset_index(drop=True)
df["label"] = df["label"].astype(int)
print(f"  {len(df)} exemplos | CT&I: {(df['label']==1).sum()} | Não-CT&I: {(df['label']==0).sum()}")

X = df["text"].values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

# ---------------------------------------------------------------------------
# 2. Embeddings
# ---------------------------------------------------------------------------
print(f"\nCarregando modelo: {EMBEDDING_MODEL}")
encoder = SentenceTransformer(EMBEDDING_MODEL)

print("Gerando embeddings...")
X_train_emb = encoder.encode(
    X_train.tolist(), normalize_embeddings=True,
    show_progress_bar=True, batch_size=64,
)
X_test_emb = encoder.encode(
    X_test.tolist(), normalize_embeddings=True,
    show_progress_bar=True, batch_size=64,
)

# ---------------------------------------------------------------------------
# 3. Balanceamento e treino
# ---------------------------------------------------------------------------
ros = RandomOverSampler(random_state=RANDOM_STATE)
X_train_bal, y_train_bal = ros.fit_resample(X_train_emb, y_train)

print("\nTreinando SVM Linear (CalibratedClassifierCV)...")
model = CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=RANDOM_STATE))
model.fit(X_train_bal, y_train_bal)

y_pred = model.predict(X_test_emb)
acc = (y_pred == y_test).mean()
print(f"  Acurácia no teste: {acc:.4f}")

# ---------------------------------------------------------------------------
# 4. Função de predição para o LIME
#    LIME passa listas de strings perturbadas; encode normalizado como no treino
# ---------------------------------------------------------------------------
def predict_proba(texts):
    embs = encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return model.predict_proba(embs)

# ---------------------------------------------------------------------------
# 5. Selecionar exemplos
# ---------------------------------------------------------------------------
results_df = pd.DataFrame({
    "text":    X_test,
    "real":    y_test,
    "predito": y_pred,
})
results_df["correto"] = results_df["real"] == results_df["predito"]

rng = np.random.default_rng(RANDOM_STATE)

def sample_n(mask, n=NUM_SAMPLES):
    idx = results_df[mask].index.tolist()
    chosen = rng.choice(idx, size=min(n, len(idx)), replace=False)
    return results_df.loc[chosen]

fp_df  = sample_n((results_df["real"] == 0) & (results_df["predito"] == 1))
fn_df  = sample_n((results_df["real"] == 1) & (results_df["predito"] == 0))
tp_df  = sample_n((results_df["real"] == 1) & (results_df["correto"] == True))
tn_df  = sample_n((results_df["real"] == 0) & (results_df["correto"] == True))

grupos = [
    ("Falso Positivo",        fp_df),
    ("Falso Negativo",        fn_df),
    ("Acerto CT&I (TP)",      tp_df),
    ("Acerto Não-CT&I (TN)",  tn_df),
]

print(f"\n  FP selecionados: {len(fp_df)} | FN: {len(fn_df)} | TP: {len(tp_df)} | TN: {len(tn_df)}")

# ---------------------------------------------------------------------------
# 6. LIME
# ---------------------------------------------------------------------------
explainer = LimeTextExplainer(class_names=["Não-CT&I", "CT&I"])

rows = []
sep = "=" * 70

for grupo, subset in grupos:
    print(f"\n{sep}")
    print(f"GRUPO: {grupo}")
    print(sep)

    for _, row in subset.iterrows():
        texto   = row["text"]
        real    = "CT&I" if row["real"] == 1 else "Não-CT&I"
        predito = "CT&I" if row["predito"] == 1 else "Não-CT&I"

        exp = explainer.explain_instance(
            texto,
            predict_proba,
            num_features=NUM_FEATURES,
            num_samples=NUM_LIME_SAMPLES,
        )

        features = exp.as_list()   # [(token, peso), ...]
        prob = exp.predict_proba   # [p(Não-CT&I), p(CT&I)]

        print(f"\n  Texto  : {texto}")
        print(f"  Real   : {real}  |  Predito: {predito}")
        print(f"  P(CT&I): {prob[1]:.3f}  |  P(Não-CT&I): {prob[0]:.3f}")
        print(f"  Top-{NUM_FEATURES} tokens LIME:")
        for token, peso in features:
            sinal = "→ CT&I" if peso > 0 else "→ Não-CT&I"
            print(f"    {token:30s}  peso={peso:+.4f}  ({sinal})")

        rows.append({
            "grupo":          grupo,
            "texto":          texto,
            "real":           real,
            "predito":        predito,
            "p_cti":          round(float(prob[1]), 4),
            "p_nao_cti":      round(float(prob[0]), 4),
            **{f"token_{i+1}": features[i][0]   if i < len(features) else ""  for i in range(NUM_FEATURES)},
            **{f"peso_{i+1}":  round(features[i][1], 4) if i < len(features) else None for i in range(NUM_FEATURES)},
        })

# ---------------------------------------------------------------------------
# 7. Salvar CSV
# ---------------------------------------------------------------------------
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
out_path = RESULTS_DIR / "lime_explanations.csv"
pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8")
print(f"\n{sep}")
print(f"Resultados salvos em: {out_path}")
