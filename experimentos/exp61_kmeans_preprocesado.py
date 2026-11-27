"""K-means, fase 2: efecto del preprocesado.

Las dos mejores configuraciones de exp60 (elegidas sobre TRAIN), que además son
las dos estrategias de selección distintas:
    - un clúster:     Cr_b, K = 7,  max_suma             (Dice train 0.596)
    - varios clústeres: Cr_b, K = 12, relativa, margen 30 (Dice train 0.588)
se prueban con los cuatro preprocesados comunes P0-P3.

Ejecutar:
    uv run python scripts/ejecutar.py exp61 --conjunto train --sin-figuras
    uv run python scripts/resumen.py --conjunto train --filtro exp61
La mejor combinación se copia a exp62_kmeans_postprocesado.py.
"""

from src.preprocesado.variantes import PREPROCESADOS

MEJORES = {
    "Cr_b_k7_max_suma": ("kmeans", {"caracteristicas": "Cr_b", "k": 7, "seleccion": "max_suma"}),
    "Cr_b_k12_relativa_m30": ("kmeans", {"caracteristicas": "Cr_b", "k": 12, "seleccion": "relativa", "margen": 30}),
}

EXPERIMENTOS = [
    {
        "nombre": f"exp61_kmeans_{id_modelo}_{id_pre}",
        "descripcion": f"K-means {id_modelo} con preprocesado {id_pre}",
        "preprocesado": pasos,
        "modelo": modelo,
    }
    for id_modelo, modelo in MEJORES.items()
    for id_pre, pasos in PREPROCESADOS.items()
]