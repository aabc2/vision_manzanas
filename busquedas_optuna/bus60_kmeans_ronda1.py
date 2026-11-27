"""K-means con Optuna (alternativa a exp60-exp62).

Optuna ajusta a la vez preprocesado, modelo y postprocesado, sobre TRAIN.
Rangos elegidos a partir de lo que enseñó la rejilla:
    - características: ab o Cr_b (bgr y lab, que incluyen el brillo, fueron
      claramente peores).
    - K: 3-16. Con un solo clúster el mejor K fue pequeño (7); con la regla
      "relativa", que elige varios, seguía mejorando al subir K.
    - selección del clúster manzana: las cuatro reglas; margen solo para "relativa".

Uso:
    uv run python scripts/optimizar.py bus60 --pruebas 150
    uv run python scripts/optimizar.py bus60 --evaluar-test     # al final
"""

from src.optimizacion import sugerir_postprocesado, sugerir_preprocesado


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
