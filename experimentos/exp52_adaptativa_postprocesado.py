"""Umbralización adaptativa, fase 3: efecto del postprocesado.

La mejor combinación de modelo + preprocesado de exp51 (elegida sobre TRAIN):
    media, Cr OR b*, bloque 301, C = -30, con P2_bilateral (Dice train 0.548)
se prueba con varios postprocesados.

Los cuatro primeros son los mismos que en Otsu (exp32), para poder comparar
los dos métodos en igualdad de condiciones. Los dos últimos prueban un
elemento estructurante mayor (7) y un área mínima mayor (200), porque la
umbralización adaptativa con OR deja manchas sueltas en zonas con textura.

Ejecutar:
    uv run python scripts/ejecutar.py exp52 --conjunto train --sin-figuras
    uv run python scripts/resumen.py --conjunto train --filtro exp52

Cuando las tres fases estén decididas, la evaluación final es sobre TEST:
    uv run python scripts/ejecutar.py exp50 exp51 exp52
    uv run python scripts/resumen.py --filtro exp5
"""

from src.preprocesado.variantes import PREPROCESADOS

MODELO = ("adaptativa_combinada", {
    "canales": ("Cr", "b"), "operacion": "or", "metodo": "media", "bloque": 301, "C": -30})
PREPROCESADO = PREPROCESADOS["P2_bilateral"]

POSTPROCESADOS = {
    # Los mismos que en exp32 (Otsu).
    "sin": [],
    "apertura5": [("apertura", {"k": 5})],
    "apertura5_area80": [("apertura", {"k": 5}), ("filtrar_area", {"minima": 80})],
    "apertura5_cierre5_area80": [("apertura", {"k": 5}), ("cierre", {"k": 5}), ("filtrar_area", {"minima": 80})],
    # Propios de la adaptativa.
    "apertura7_cierre7_area80": [("apertura", {"k": 7}), ("cierre", {"k": 7}), ("filtrar_area", {"minima": 80})],
    "apertura5_cierre5_area200": [("apertura", {"k": 5}), ("cierre", {"k": 5}), ("filtrar_area", {"minima": 200})],
}

EXPERIMENTOS = [
    {
        "nombre": f"exp52_adapt_post_{id_post}",
        "descripcion": f"Mejor adaptativa con postprocesado {id_post}",
        "preprocesado": PREPROCESADO,
        "modelo": MODELO,
        "postprocesado": pasos,
    }
    for id_post, pasos in POSTPROCESADOS.items()
]