"""Umbralización adaptativa con Optuna (alternativa a exp50-exp52).

Optuna ajusta a la vez preprocesado, modelo y postprocesado, sobre TRAIN.
Rangos elegidos a partir de lo que enseñó la rejilla:
    - canal: Cr, b* o sus combinaciones (gris descartado: fue el peor).
    - bloque: 51-501, impar. Tiene que ser mayor que una manzana.
    - C: -50 a 0. Negativo = el píxel tiene que superar la media de su zona.
    - k: solo para Niblack.

Uso:
    uv run python scripts/optimizar.py bus50 --pruebas 150
    uv run python scripts/optimizar.py bus50 --evaluar-test     # al final
"""

from src.optimizacion import sugerir_postprocesado, sugerir_preprocesado

CANALES = {
    "Cr": ("adaptativa", {"canal": "Cr"}),
    "b": ("adaptativa", {"canal": "b"}),
    "Cr_and_b": ("adaptativa_combinada", {"canales": ("Cr", "b"), "operacion": "and"}),
    "Cr_or_b": ("adaptativa_combinada", {"canales": ("Cr", "b"), "operacion": "or"}),
}


def espacio(trial):
    modelo, params = CANALES[trial.suggest_categorical("canal", list(CANALES))]
    params = dict(params)
    params["metodo"] = trial.suggest_categorical("metodo", ["media", "gaussiana", "niblack"])
    params["bloque"] = trial.suggest_int("bloque", 51, 501, step=2)
    params["C"] = trial.suggest_int("C", -50, 0)
    if params["metodo"] == "niblack":
        params["k"] = trial.suggest_float("k", 0.0, 1.5)
    return {
        "preprocesado": sugerir_preprocesado(trial),
        "modelo": (modelo, params),
        "postprocesado": sugerir_postprocesado(trial),
    }
