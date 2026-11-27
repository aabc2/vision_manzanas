"""Búsqueda de hiperparámetros con Optuna (alternativa a la rejilla de experimentos/).

Idea:
    En la rejilla (experimentos/expXX) se prueban TODAS las combinaciones que
    escribimos a mano. Optuna prueba combinaciones una a una y elige la
    siguiente según los resultados anteriores (muestreador TPE): prueba más
    cerca de lo que salió bien y menos donde salió mal. Así puede buscar en
    rangos continuos (C entre -50 y 0, bloque entre 51 y 501...) y ajustar a la
    vez preprocesado, modelo y postprocesado, en vez de fase por fase.

Cada búsqueda es un archivo busquedas_optuna/busXX_*.py con una función
    espacio(trial) -> {"preprocesado": [...], "modelo": (nombre, {params}),
                       "postprocesado": [...], "escala": 1.0}
que devuelve un experimento con el mismo formato que src/evaluacion.py.
El preprocesado y el postprocesado se eligen con sugerir_preprocesado() y
sugerir_postprocesado(), comunes a todo el grupo, para que todos los métodos
se comparen con el mismo espacio de búsqueda.

Validación cruzada (lo que pidió el profesor):
    - Optuna solo ve las 100 imágenes de TRAIN, repartidas en 5 folds fijos
      (src/particion.json). Cada combinación se evalúa en los 5 folds y su
      nota es la MEDIA del Dice de los 5 folds. También se guarda la
      desviación típica entre folds: si es pequeña, la combinación funciona
      igual de bien con cualquier grupo de imágenes (es estable).
    - Como estos métodos no aprenden de las imágenes (no se entrenan), en
      cada fold no hay nada que entrenar: solo se evalúa. Lo que se "ajusta"
      con los datos son los hiperparámetros, y se eligen con esa media.
    - El test del profesor (data/test) se usa una sola vez al final, con
      scripts/optimizar.py <busqueda> --evaluar-test.
    - Se maximiza el Dice (la exactitud no sirve: el fondo es ~95 %).

Eficiencia:
    - Las imágenes de train se cargan una sola vez en memoria, y cada
      preprocesado se calcula una sola vez y se reutiliza (caché).
    - Poda (MedianPruner): al terminar cada fold se informa del Dice medio de
      los folds hechos; si una prueba va peor que la mediana de las
      anteriores en ese punto, se abandona sin evaluar el resto de folds.
"""

import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna

from src import datos, metricas, modelos, postprocesado
from src.evaluacion import PROTOCOLO, resumen_folds
from src.preprocesado import aplicar as preprocesar
from src.preprocesado import describir as describir_pre
from src.preprocesado import redimensionar
from src.preprocesado.variantes import PREPROCESADOS

DIR_OPTUNA = datos.RAIZ / "resultados" / "optuna"
CLAVES_ATRIBUTOS = ("std_dice_folds", "media_iou", "media_precision", "media_recall", "std_dice", "tiempo_medio_s")
DIR_MEJORES = datos.RAIZ / "busquedas_optuna" / "mejores"
RUTA_BD = DIR_OPTUNA / "estudios.db"


# --------------------------------------------------------------------------- #
# Espacio de búsqueda común: preprocesado y postprocesado
# --------------------------------------------------------------------------- #

def sugerir_preprocesado(trial):
    """Uno de los preprocesados comunes P0-P3 (nunca escritos a mano)."""
    nombre = trial.suggest_categorical("preprocesado", list(PREPROCESADOS))
    return PREPROCESADOS[nombre]


def sugerir_postprocesado(trial):
    """Apertura, cierre y filtro de área opcionales (0 = no se aplica).

    Los tamaños del elemento estructurante son impares (3-9) y el área mínima
    va de 0 a 400 píxeles en pasos de 20.
    """
    pasos = []
    k_apertura = trial.suggest_categorical("post_apertura", [0, 3, 5, 7, 9])
    k_cierre = trial.suggest_categorical("post_cierre", [0, 3, 5, 7, 9])
    area = trial.suggest_int("post_area", 0, 400, step=20)
    if k_apertura:
        pasos.append(("apertura", {"k": k_apertura}))
    if k_cierre:
        pasos.append(("cierre", {"k": k_cierre}))
    if area:
        pasos.append(("filtrar_area", {"minima": area}))
    return pasos


# --------------------------------------------------------------------------- #
# Evaluación rápida sobre train
# --------------------------------------------------------------------------- #

class DatosEnMemoria:
    """Imágenes y máscaras de un conjunto, cargadas una vez, con caché de preprocesados."""

    def __init__(self, verbose=True):
        inicio = time.perf_counter()
        # Ordenadas por fold: primero las del fold 0, luego las del 1... para
        # poder informar a Optuna al terminar cada fold.
        self.folds = datos.folds()
        self.nombres = [n for f in self.folds for n in f]
        self.fold = [i for i, f in enumerate(self.folds) for _ in f]
        parejas = [datos.cargar_pareja(n) for n in self.nombres]
        self.imagenes = [p[0] for p in parejas]
        self.mascaras = [p[1] for p in parejas]
        self._cache = {}
        if verbose:
            print(f"{len(self.nombres)} imágenes de train ({len(self.folds)} folds de "
                  f"{', '.join(str(len(f)) for f in self.folds)}) cargadas en {time.perf_counter() - inicio:.1f} s")

    def preprocesadas(self, pasos):
        clave = describir_pre(pasos)
        if clave not in self._cache:
            self._cache[clave] = [preprocesar(img, pasos) for img in self.imagenes]
        return self._cache[clave]


def evaluar(config, datos_mem, trial=None):
    """Métricas de un experimento con validación cruzada sobre las imágenes en memoria.

    Devuelve las medias por imagen (media_dice, media_iou...) y las de los
    folds (dice_por_fold, media_dice_folds, std_dice_folds).
    Si se pasa `trial`, al terminar cada fold informa del Dice medio de los
    folds hechos y lanza optuna.TrialPruned si Optuna decide abandonar.
    """
    modelo, params = config["modelo"]
    if modelos.es_entrenable(modelo):
        raise ValueError("la búsqueda con Optuna es para métodos sin entrenamiento")
    segmentar = modelos.preparar(modelo, params)
    pasos_post = config.get("postprocesado", [])
    escala = config.get("escala", 1.0)

    filas = []
    inicio = time.perf_counter()
    for i, (entrada, referencia) in enumerate(zip(datos_mem.preprocesadas(config["preprocesado"]),
                                                  datos_mem.mascaras), 1):
        alto, ancho = referencia.shape
        if escala < 1:
            entrada = redimensionar(entrada, escala)
        prediccion = np.where(segmentar(entrada) > 0, 255, 0).astype(np.uint8)
        if prediccion.shape != (alto, ancho):
            prediccion = redimensionar(prediccion, tamano=(ancho, alto), es_mascara=True)
        prediccion = postprocesado.aplicar(prediccion, pasos_post)
        fila = metricas.calcular(prediccion, referencia)
        fila["fold"] = datos_mem.fold[i - 1]
        filas.append(fila)

        fin_de_fold = i == len(datos_mem.fold) or datos_mem.fold[i] != fila["fold"]
        if trial is not None and fin_de_fold:
            parcial = resumen_folds(filas)["media_dice_folds"]
            trial.report(parcial, step=fila["fold"] + 1)
            if trial.should_prune():
                raise optuna.TrialPruned()

    resumen = metricas.agregar(filas)
    resumen.update(resumen_folds(filas))
    resumen["tiempo_medio_s"] = (time.perf_counter() - inicio) / len(filas)
    return resumen


# --------------------------------------------------------------------------- #
# Estudio de Optuna
# --------------------------------------------------------------------------- #

def crear_estudio(nombre, semilla=0, podar=True, nuevo=False):
    """Estudio guardado en resultados/optuna/estudios.db (se puede reanudar)."""
    DIR_OPTUNA.mkdir(parents=True, exist_ok=True)
    almacen = f"sqlite:///{RUTA_BD.as_posix()}"
    if nuevo:
        try:
            optuna.delete_study(study_name=nombre, storage=almacen)
        except KeyError:
            pass
    estudio = optuna.create_study(
        study_name=nombre,
        storage=almacen,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=semilla),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1) if podar
        else optuna.pruners.NopPruner(),
    )
    # Un estudio hecho con la partición antigua (70/30) no se puede continuar con
    # la validación cruzada: sus notas no son comparables.
    if estudio.trials and estudio.user_attrs.get("protocolo") != PROTOCOLO:
        raise RuntimeError(
            f"el estudio '{nombre}' se hizo con la partición antigua (70 train / 30 test). "
            f"Empieza de cero con: uv run python scripts/optimizar.py {nombre} --nuevo"
        )
    estudio.set_user_attr("protocolo", PROTOCOLO)
    return estudio


def objetivo(espacio, datos_mem):
    """Función que Optuna maximiza: la media del Dice de los 5 folds."""
    def funcion(trial):
        config = espacio(trial)
        resumen = evaluar(config, datos_mem, trial=trial)
        for clave in CLAVES_ATRIBUTOS:
            if clave in resumen:
                trial.set_user_attr(clave, float(resumen[clave]))
        trial.set_user_attr("dice_por_fold", [round(d, 4) for d in resumen["dice_por_fold"]])
        trial.set_user_attr("config", json.dumps(config, default=str, ensure_ascii=False))
        return float(resumen["media_dice_folds"])
    return funcion


def config_de_prueba(trial):
    """Recupera el experimento (preprocesado, modelo, postprocesado) de una prueba."""
    config = json.loads(trial.user_attrs["config"])
    config["modelo"] = tuple(config["modelo"])
    config["preprocesado"] = [tuple(p) for p in config["preprocesado"]]
    config["postprocesado"] = [tuple(p) for p in config.get("postprocesado", [])]
    return config


# --------------------------------------------------------------------------- #
# Resultados: CSV, mejor configuración y figuras
# --------------------------------------------------------------------------- #

def guardar_resultados(estudio):
    """Guarda pruebas.csv, mejor.json y las figuras en resultados/optuna/<nombre>/,
    y copia la mejor configuración a busquedas_optuna/mejores/<nombre>.json (sí se sube a Git)."""
    nombre = estudio.study_name
    carpeta = DIR_OPTUNA / nombre
    carpeta.mkdir(parents=True, exist_ok=True)
    pruebas = estudio.get_trials(deepcopy=False)
    completas = [t for t in pruebas if t.state == optuna.trial.TrialState.COMPLETE]
    if not completas:
        raise RuntimeError("ninguna prueba ha terminado; no hay nada que guardar")

    # 1. Todas las pruebas, ordenadas por Dice.
    claves_param = sorted({k for t in pruebas for k in t.params})
    claves_attr = list(CLAVES_ATRIBUTOS)
    with open(carpeta / "pruebas.csv", "w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        escritor.writerow(["prueba", "estado", "dice_cv"] + claves_attr + ["dice_por_fold"] + claves_param)
        for t in sorted(pruebas, key=lambda t: -(t.value if t.value is not None else -1)):
            escritor.writerow([t.number, t.state.name, t.value]
                              + [t.user_attrs.get(k) for k in claves_attr]
                              + [" ".join(str(d) for d in t.user_attrs.get("dice_por_fold", []))]
                              + [t.params.get(k) for k in claves_param])

    # 2. La mejor configuración.
    mejor = estudio.best_trial
    salida = {
        "busqueda": nombre,
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "validacion": f"validación cruzada, {len(datos.leer_particion()['folds'])} folds sobre train",
        "pruebas_completas": len(completas),
        "pruebas_podadas": sum(t.state == optuna.trial.TrialState.PRUNED for t in pruebas),
        "mejor_prueba": mejor.number,
        "dice_cv": mejor.value,
        "dice_por_fold": mejor.user_attrs.get("dice_por_fold"),
        "metricas_cv": {k: mejor.user_attrs.get(k) for k in claves_attr},
        "parametros": mejor.params,
        "config": config_de_prueba(mejor),
    }
    for ruta in (carpeta / "mejor.json", DIR_MEJORES / f"{nombre}.json"):
        ruta.parent.mkdir(parents=True, exist_ok=True)
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(salida, f, indent=2, ensure_ascii=False, default=str)

    # 3. Figuras para la memoria.
    _figura_historia(pruebas, carpeta / "historia.png", nombre)
    _figura_parametros(completas, claves_param, carpeta / "parametros.png", nombre)
    _figura_importancia(estudio, carpeta / "importancia.png", nombre)
    return salida


def _figura_historia(pruebas, ruta, titulo):
    """Dice de cada prueba y mejor Dice acumulado: cómo aprende Optuna."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, eje = plt.subplots(figsize=(8, 4))
    completas = [t for t in pruebas if t.state == optuna.trial.TrialState.COMPLETE]
    podadas = [t for t in pruebas if t.state == optuna.trial.TrialState.PRUNED]
    if podadas:
        eje.scatter([t.number for t in podadas], [t.intermediate_values[max(t.intermediate_values)]
                                                  if t.intermediate_values else 0 for t in podadas],
                    s=12, color="0.75", label="podada (Dice parcial)")
    eje.scatter([t.number for t in completas], [t.value for t in completas], s=14, label="completa")
    numeros = sorted(t.number for t in completas)
    valores = {t.number: t.value for t in completas}
    acumulado = np.maximum.accumulate([valores[n] for n in numeros])
    eje.step(numeros, acumulado, where="post", color="C3", label="mejor hasta ahora")
    eje.set_xlabel("prueba")
    eje.set_ylabel("Dice (media de los 5 folds)")
    eje.set_title(f"{titulo}: historia de la búsqueda")
    eje.legend()
    fig.tight_layout()
    fig.savefig(ruta, dpi=110)
    plt.close(fig)


def _figura_parametros(completas, claves, ruta, titulo):
    """Dice frente a cada parámetro (sustituye a las tablas de la rejilla)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columnas = 3
    filas = int(np.ceil(len(claves) / columnas))
    fig, ejes = plt.subplots(filas, columnas, figsize=(4 * columnas, 3 * filas), squeeze=False)
    rng = np.random.default_rng(0)
    for eje, clave in zip(ejes.flat, claves):
        puntos = [(t.params[clave], t.value) for t in completas if clave in t.params]
        valores = [p[0] for p in puntos]
        dices = [p[1] for p in puntos]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in valores):
            eje.scatter(valores, dices, s=10, alpha=0.6)
        else:  # categórico: una columna por categoría, con un poco de dispersión horizontal
            categorias = sorted({str(v) for v in valores})
            x = [categorias.index(str(v)) + rng.uniform(-0.15, 0.15) for v in valores]
            eje.scatter(x, dices, s=10, alpha=0.6)
            eje.set_xticks(range(len(categorias)), categorias, rotation=30, ha="right", fontsize=8)
        eje.set_title(clave, fontsize=10)
        eje.set_ylabel("Dice")
    for eje in list(ejes.flat)[len(claves):]:
        eje.axis("off")
    fig.suptitle(f"{titulo}: Dice frente a cada parámetro")
    fig.tight_layout()
    fig.savefig(ruta, dpi=110)
    plt.close(fig)


def _figura_importancia(estudio, ruta, titulo):
    """Qué parámetros influyen más en el Dice (fANOVA de Optuna).

    Solo entran los parámetros que tienen TODAS las pruebas. Los condicionales
    (k de Niblack, margen de K-means) se dejan fuera: Optuna calcularía su
    importancia solo con las pruebas que los usan, y el gráfico engañaría.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    completas = estudio.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    comunes = sorted(set.intersection(*(set(t.params) for t in completas))) if completas else []
    try:
        importancias = optuna.importance.get_param_importances(estudio, params=comunes)
    except (RuntimeError, ValueError):
        return  # hacen falta varias pruebas completas
    nombres = list(importancias)[::-1]
    fig, eje = plt.subplots(figsize=(7, 0.4 * len(nombres) + 1.2))
    eje.barh(nombres, [importancias[n] for n in nombres])
    eje.set_xlabel("importancia relativa")
    eje.set_title(f"{titulo}: importancia de cada parámetro")
    fig.tight_layout()
    fig.savefig(ruta, dpi=110)
    plt.close(fig)


def cargar_mejor(nombre):
    """Lee busquedas_optuna/mejores/<nombre>.json (la mejor configuración encontrada)."""
    ruta = DIR_MEJORES / f"{nombre}.json"
    if not ruta.exists():
        raise FileNotFoundError(f"no existe {ruta.relative_to(datos.RAIZ)}. "
                                f"Lanza antes: uv run python scripts/optimizar.py {nombre}")
    with open(ruta, encoding="utf-8") as f:
        salida = json.load(f)
    config = salida["config"]
    config["modelo"] = tuple(config["modelo"])
    config["preprocesado"] = [tuple(p) for p in config["preprocesado"]]
    config["postprocesado"] = [tuple(p) for p in config.get("postprocesado", [])]
    return salida, config
