"""Umbralización adaptativa con Optuna, segunda ronda (bus50 era la primera).

Motivo: en bus50 el mejor cierre salió en el borde del rango (9, el máximo),
así que aquí se amplía la zona de búsqueda:
    - cierre hasta 15 (y apertura hasta 15, por simetría),
    - bloque hasta 701 (el mejor fue 391, lejos del borde, pero por si acaso).
El resto es igual que bus50. Se descartan b y Cr solos, que fueron peores.

Empieza probando lo mejor de la rejilla (exp52) y lo mejor de bus50
(PUNTOS_DE_PARTIDA), así que nunca acaba por debajo de ellos en train.

El postprocesado se define aquí y no con sugerir_postprocesado(), para no
cambiar el espacio común que usan los demás.

Uso:
    uv run python scripts/optimizar.py bus51 --pruebas 150
    uv run python scripts/optimizar.py bus51 --evaluar-test     # al final, una vez
"""

from src.optimizacion import sugerir_preprocesado

CANALES = {
    "Cr_and_b": ("adaptativa_combinada", {"canales": ("Cr", "b"), "operacion": "and"}),
    "Cr_or_b": ("adaptativa_combinada", {"canales": ("Cr", "b"), "operacion": "or"}),
}

PUNTOS_DE_PARTIDA = [
    # Mejor de la rejilla (exp52_adapt_post_apertura5_cierre5_area80): Dice train 0.555
    {"canal": "Cr_or_b", "metodo": "media", "bloque": 301, "C": -30, "preprocesado": "P2_bilateral",
     "post_apertura": 5, "post_cierre": 5, "post_area": 80},
    # Mejor de la primera ronda de Optuna (bus50, prueba 122): Dice train 0.574
    {"canal": "Cr_or_b", "metodo": "media", "bloque": 391, "C": -29, "preprocesado": "P2_bilateral",
     "post_apertura": 3, "post_cierre": 9, "post_area": 140},
]


def espacio(trial):
    modelo, params = CANALES[trial.suggest_categorical("canal", list(CANALES))]
    params = dict(params)
    params["metodo"] = trial.suggest_categorical("metodo", ["media", "gaussiana", "niblack"])
    params["bloque"] = trial.suggest_int("bloque", 51, 701, step=2)
    params["C"] = trial.suggest_int("C", -50, 0)
    if params["metodo"] == "niblack":
        params["k"] = trial.suggest_float("k", 0.0, 1.5)

    post = []
    k_apertura = trial.suggest_categorical("post_apertura", [0, 3, 5, 7, 9, 11, 13, 15])
    k_cierre = trial.suggest_categorical("post_cierre", [0, 3, 5, 7, 9, 11, 13, 15])
    area = trial.suggest_int("post_area", 0, 400, step=20)
    if k_apertura:
        post.append(("apertura", {"k": k_apertura}))
    if k_cierre:
        post.append(("cierre", {"k": k_cierre}))
    if area:
        post.append(("filtrar_area", {"minima": area}))

    return {
        "preprocesado": sugerir_preprocesado(trial),
        "modelo": (modelo, params),
        "postprocesado": post,
    }
