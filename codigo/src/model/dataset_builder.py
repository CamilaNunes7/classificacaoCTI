"""
Constrói o dataset unificado para o classificador CT&I.

Saída: data/output/ml/dataset.csv
Colunas: text, label (1=CT&I, 0=não-CT&I), source, domain
"""

import json
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ANNOTATIONS_DIR = BASE_DIR / "data" / "input" / "annotations"
NEGATIVE_FILE = BASE_DIR / "data" / "input" / "negative_examples" / "negative_indicators.json"
OUTPUT_DIR = BASE_DIR / "data" / "output" / "ml"


def build_dataset() -> pd.DataFrame:
    rows = []

    # Classe positiva: CT&I
    for txt_path in sorted(ANNOTATIONS_DIR.glob("*.txt")):
        for line in txt_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                rows.append({
                    "text": line,
                    "label": 1,
                    "source": txt_path.name,
                    "domain": "cti",
                })

    # Classe negativa: não-CT&I
    negatives = json.loads(NEGATIVE_FILE.read_text(encoding="utf-8"))
    for item in negatives:
        text = item.get("text", "").strip()
        if text:
            rows.append({
                "text": text,
                "label": 0,
                "source": item.get("source_pdf", "legado"),
                "domain": item.get("domain", "desconhecido"),
            })

    df = pd.DataFrame(rows).drop_duplicates(subset="text").reset_index(drop=True)
    return df


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_dataset()
    out = OUTPUT_DIR / "dataset.csv"
    df.to_csv(out, index=False, encoding="utf-8")

    print(f"Dataset salvo em: {out}")
    print(f"Total: {len(df)} exemplos")
    print(df["label"].value_counts().rename({1: "CT&I (1)", 0: "Não-CT&I (0)"}).to_string())
    print(f"\nDomínios (negativos):")
    print(df[df.label == 0]["domain"].value_counts().to_string())
