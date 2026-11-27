"""K-means con Optuna, segunda ronda (bus60 era la primera).

Motivo: en bus60, con 100 pruebas (60 completas), Optuna no llegó a la mejor
combinación de la rejilla (Dice train 0.605) aunque estaba dentro de su zona
de búsqueda: se quedó en max_Cr con K = 6 (0.584). El problema no era la zona,
sino el número de pruebas, así que:
    - mismo espacio de búsqueda que bus60,
    - más pruebas (300),
    - empieza probando lo mejor de la rejilla y lo mejor de bus60
      (PUNTOS_DE_PARTIDA), así que nunca acaba por debajo de ellos en train.

Uso:
    uv run python scripts/optimizar.py bus61 --pruebas 300
    uv run python scripts/optimizar.py bus61 --evaluar-test     # al final, una vez
"""

from src.optimizacion import sugerir_postprocesado, sugerir_preprocesado

PUNTOS_DE_PARTIDA = [
    # Mejor de la rejilla (exp62_kmeans_post_sin): Dice train 0.605
    {"caracteristicas": "Cr_b", "k": 12, "seleccion": "relativa", "margen": 30.0,
     "preprocesado": "P2_bilateral", "post_apertura": 0, "post_cierre": 0, "post_area": 0},
    # Mejor de la primera ronda de Optuna (bus60, prueba 97): Dice train 0.584
    {"caracteristicas": "Cr_b", "k": 6, "seleccion": "max_Cr",
     "preprocesado": "P2_bilateral", "post_apertura": 0, "post_cierre": 9, "post_area": 20},
]


def espacio(trial):
    params = {
        "caracteristicas": trial.suggest_categorical("caracteristicas", ["ab", "Cr_b"]),
        "k": trial.suggest_int("k", 3, 16),
        "seleccion": trial.suggest_categorical("seleccion", ["max_Cr", "max_suma", "max_Cr_o_max_b", "relativa"]),
    }
    if params["seleccion"] == "relativa":
        params["margen"] = trial.suggest_float("margen", 10.0, 50.0)
    return {
        "preprocesado": sugerir_preprocesado(trial),
        "modelo": ("kmeans", params),
        "postprocesado": sugerir_postprocesado(trial),
    }
