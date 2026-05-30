from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class IndicatorOut(BaseModel):
    id: str
    name: str
    area: str
    subarea: str
    keywords: List[str]
    source_article: str
    unit: str
    excerpt: str
    language: str
    annotated_by: str
    definition: str
    validated: bool


class DatasetPage(BaseModel):
    total: int
    offset: int
    limit: int
    indicators: List[IndicatorOut]


class SubareaCount(BaseModel):
    subarea: str
    count: int


class AreaStats(BaseModel):
    area: str
    count: int
    subareas: List[SubareaCount]


class DatasetStats(BaseModel):
    total_indicators: int
    total_articles: int
    enriched: bool
    areas: List[AreaStats]
    by_annotator: Dict[str, int]
    by_language: Dict[str, int]


class ArticleSummary(BaseModel):
    id: str
    human_count: int
    gemini_count: Optional[int]
    has_pdf: bool
    has_comparison: bool


class ExtractionOut(BaseModel):
    filename: str
    model: str
    indicator_count: int
    indicators: List[Dict[str, Any]]


class AreaMetricRow(BaseModel):
    area: str
    ref_count: int
    hyp_count: int
    precision: float
    recall: float
    f1: float


class AreaMetricsOut(BaseModel):
    enriched: bool
    message: Optional[str] = None
    macro_by_area: List[AreaMetricRow]
    per_article: Optional[Dict[str, Any]] = None


class EnrichStatus(BaseModel):
    status: str
    message: str
