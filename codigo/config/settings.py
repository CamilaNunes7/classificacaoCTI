import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# Caminhos de entrada
INPUT_ARTICLES_DIR = BASE_DIR / "data" / "input" / "articles"
INPUT_ANNOTATIONS_DIR = BASE_DIR / "data" / "input" / "annotations"

# Caminhos de saída
OUTPUT_EXTRACTED_DIR = BASE_DIR / "data" / "output" / "extracted"
OUTPUT_COMPARISON_DIR = BASE_DIR / "data" / "output" / "comparison"
OUTPUT_REPORTS_DIR = BASE_DIR / "data" / "output" / "reports"
OUTPUT_PROVENANCE_DIR = BASE_DIR / "data" / "output" / "provenance"
OUTPUT_DATASET_DIR = BASE_DIR / "data" / "output" / "dataset"

# Modelos de IA
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_STRUCTURED_MODEL = "gemini-2.5-pro"  # Para extração estruturada e enriquecimento
CLAUDE_MODEL = "claude-opus-4-6"

# Chaves de API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Thresholds de comparação
FUZZY_THRESHOLD = 80       # Score WRatio de 0 a 100
SEMANTIC_THRESHOLD = 0.80  # Similaridade de cosseno de 0 a 1
BERTSCORE_THRESHOLD = 0.85 # BERTScore F1 token-level de 0 a 1

# Modelo de embeddings (multilíngue PT+EN, offline, sem custo de API)
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"

# Modelo para BERTScore (reutiliza o mesmo já em cache do semantic matcher)
BERTSCORE_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Configurações de retry para chamadas de API
API_MAX_RETRIES = 3
API_RETRY_DELAY_SECONDS = 10
