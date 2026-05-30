from dataclasses import dataclass, field
from typing import List

# Taxonomia baseada no Excel de referência (dataset_indicadores_CTI.xlsx)
TAXONOMY: dict = {
    "Recursos Aplicados": ["P&D", "Financiamento público"],
    "Recursos Humanos": ["Capital humano em C&T", "Formação de RH em C&T"],
    "Produção Científica": ["Publicações", "Impacto científico"],
    "Propriedade Intelectual": ["Patentes", "Marcas e outros"],
    "Inovação": ["Inovação empresarial", "Adoção tecnológica"],
    "Infraestrutura de C&T": ["Laboratórios e institutos", "Parques tecnológicos"],
    "Transferência de Tecnologia": ["Licenciamentos", "Spin-offs", "Cooperação P&D"],
    "Internacionalização": ["Colaborações internacionais", "Exportações tecnológicas"],
}

AREAS = list(TAXONOMY.keys())


@dataclass
class Indicator:
    id: str
    name: str
    area: str
    subarea: str
    keywords: List[str]
    source_article: str
    unit: str = "N/A"
    excerpt: str = ""
    language: str = "pt"
    annotated_by: str = "human"
    definition: str = ""
    validated: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "area": self.area,
            "subarea": self.subarea,
            "keywords": self.keywords,
            "source_article": self.source_article,
            "unit": self.unit,
            "excerpt": self.excerpt,
            "language": self.language,
            "annotated_by": self.annotated_by,
            "definition": self.definition,
            "validated": self.validated,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Indicator":
        valid = {
            "id", "name", "area", "subarea", "keywords", "source_article",
            "unit", "excerpt", "language", "annotated_by", "definition", "validated",
        }
        return cls(**{k: v for k, v in d.items() if k in valid})
