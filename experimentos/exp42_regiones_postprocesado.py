"""Crecimiento de regiones, fase 3: efecto del postprocesado.

La mejor combinación de modelo + preprocesado de exp41 (elegida sobre TRAIN)
se prueba con varios postprocesados (los mismos que en Otsu, para comparar).

>>> Actualiza MODELO y PREPROCESADO con lo que salga de exp41 en train. <<<

Ejecutar:
    uv run python scripts/ejecutar.py exp42 --conjunto train --sin-figuras
    uv run python scripts/resumen.py --conjunto train --filtro exp42

Cuando las tres fases estén decididas, la evaluación final es sobre TEST:
    uv run python scripts/ejecutar.py exp40 exp41 exp42
    uv run python scripts/resumen.py --filtro exp4
"""

from src.preprocesado.variantes import PREPROCESADOS

MODELO = ("crecimiento", {"color": "H", "tolerancia": 3, "iteraciones": 25, "geometrico": True, "factor": 1.0})
PREPROCESADO = PREPROCESADOS["P0_sin"]

POSTPROCESADOS = {
    "sin": [],
    "apertura5": [("apertura", {"k": 5})],
    "apertura5_area80": [("apertura", {"k": 5}), ("filtrar_area", {"minima": 80})],
    "apertura5_cierre5_area80": [("apertura", {"k": 5}), ("cierre", {"k": 5}), ("filtrar_area", {"minima": 80})],
}

EXPERIMENTOS = [
    {
        "nombre": f"exp42_reg_post_{id_post}",
        "descripcion": f"Mejor crecimiento de regiones con postprocesado {id_post}",
        "preprocesado": PREPROCESADO,
        "modelo": MODELO,
        "postprocesado": pasos,
    }
    for id_post, pasos in POSTPROCESADOS.items()
]
