import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


def save_extraction(
    output_path: Path,
    article_id: str,
    model_name: str,
    indicators: List[str],
    error: Optional[str] = None,
) -> None:
    """Salva os indicadores extraídos por um modelo em um arquivo JSON."""
    data = {
        "article_id": article_id,
        "model": model_name,
        "indicator_count": len(indicators),
        "indicators": indicators,
    }
    if error:
        data["error"] = error

    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.debug(f"Extração salva em: {output_path}")


def load_extraction(json_path: Path) -> List[str]:
    """Carrega indicadores de um JSON de extração previamente salvo."""
    if not json_path.exists():
        log.warning(f"Arquivo de extração não encontrado: {json_path}")
        return []
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return data.get("indicators", [])


def extraction_cache_valid(json_path: Path) -> bool:
    """Retorna True se o cache existe, não contém erro e tem ao menos um indicador."""
    if not json_path.exists():
        return False
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        return "error" not in data and len(data.get("indicators", [])) > 0
    except Exception:
        return False


def save_comparison(output_path: Path, comparison_result: Dict[str, Any]) -> None:
    """Salva o resultado de comparação de um artigo em JSON."""
    output_path.write_text(
        json.dumps(comparison_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.debug(f"Comparação salva em: {output_path}")
