"""Otsu, fase 3: efecto del postprocesado.

La mejor combinación de modelo + preprocesado de exp31 (elegida sobre TRAIN)
se prueba con varios postprocesados.

>>> Actualiza MODELO y PREPROCESADO con lo que salga de exp31 en train. <<<

Ejecutar:
    uv run python scripts/ejecutar.py exp32 --conjunto train --sin-figuras
    uv run python scripts/resumen.py --conjunto train --filtro exp32

Cuando las tres fases estén decididas, la evaluación final es sobre TEST:
    uv run python scripts/ejecutar.py exp30 exp31 exp32
    uv run python scripts/resumen.py --filtro exp3
"""

from src.preprocesado.variantes import PREPROCESADOS

MODELO = ("otsu_combinado", {"canales": ("Cr", "b"), "operacion": "and", "variante": "multi"})
PREPROCESADO = PREPROCESADOS["P1_gaussiano"]

POSTPROCESADOS = {
    "sin": [],
    "apertura5": [("apertura", {"k": 5})],
    "apertura5_area80": [("apertura", {"k": 5}), ("filtrar_area", {"minima": 80})],
    "apertura5_cierre5_area80": [("apertura", {"k": 5}), ("cierre", {"k": 5}), ("filtrar_area", {"minima": 80})],
}

EXPERIMENTOS = [
    {
        "nombre": f"exp32_otsu_post_{id_post}",
        "descripcion": f"Mejor Otsu con postprocesado {id_post}",
        "preprocesado": PREPROCESADO,
        "modelo": MODELO,
        "postprocesado": pasos,
    }
    for id_post, pasos in POSTPROCESADOS.items()
]
