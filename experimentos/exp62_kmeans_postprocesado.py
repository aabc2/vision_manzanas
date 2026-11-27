"""K-means, fase 3: efecto del postprocesado.

La mejor combinación de modelo + preprocesado de exp61 (elegida sobre TRAIN):
    Cr_b, K = 12, selección relativa con margen 30, con P2_bilateral (Dice train 0.605)
se prueba con varios postprocesados.

Son los mismos seis que en la umbralización adaptativa (exp52): los cuatro de
Otsu (exp32) y dos con elemento estructurante o área mínima mayores. Así los
tres métodos se comparan en igualdad de condiciones.

Ejecutar:
    uv run python scripts/ejecutar.py exp62 --conjunto train --sin-figuras
    uv run python scripts/resumen.py --conjunto train --filtro exp62

Cuando las tres fases estén decididas, la evaluación final es sobre TEST:
    uv run python scripts/ejecutar.py exp60 exp61 --sin-figuras
    uv run python scripts/ejecutar.py exp62
    uv run python scripts/resumen.py --filtro exp6
"""

from src.preprocesado.variantes import PREPROCESADOS

MODELO = ("kmeans", {"caracteristicas": "Cr_b", "k": 12, "seleccion": "relativa", "margen": 30})
PREPROCESADO = PREPROCESADOS["P2_bilateral"]

POSTPROCESADOS = {
    # Los mismos que en exp32 (Otsu).
    "sin": [],
    "apertura5": [("apertura", {"k": 5})],
    "apertura5_area80": [("apertura", {"k": 5}), ("filtrar_area", {"minima": 80})],
    "apertura5_cierre5_area80": [("apertura", {"k": 5}), ("cierre", {"k": 5}), ("filtrar_area", {"minima": 80})],
    # Los dos extra de la adaptativa (exp52).
    "apertura7_cierre7_area80": [("apertura", {"k": 7}), ("cierre", {"k": 7}), ("filtrar_area", {"minima": 80})],
    "apertura5_cierre5_area200": [("apertura", {"k": 5}), ("cierre", {"k": 5}), ("filtrar_area", {"minima": 200})],
}

EXPERIMENTOS = [
    {
        "nombre": f"exp62_kmeans_post_{id_post}",
        "descripcion": f"Mejor K-means con postprocesado {id_post}",
        "preprocesado": PREPROCESADO,
        "modelo": MODELO,
        "postprocesado": pasos,
    }
    for id_post, pasos in POSTPROCESADOS.items()
]