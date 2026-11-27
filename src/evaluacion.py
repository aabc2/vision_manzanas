"""Ejecuta un experimento completo y guarda sus resultados.

Un experimento es un diccionario con esta forma:

    {
        "nombre": "exp01_gris_otsu_P0_sin",          # carpeta en resultados/ (única)
        "descripcion": "Línea base sin preprocesado", # opcional, para la memoria
        "preprocesado": [("gaussiano", {"k": 5})],    # pasos de src.preprocesado ([] = ninguno)
        "modelo": ("gris_otsu", {}),                  # nombre en src.modelos y sus parámetros
        "postprocesado": [("apertura", {"k": 5})],    # opcional, pasos de src.postprocesado
        "escala": 1.0,                                # opcional, < 1 para acelerar modelos lentos
    }

Para cada imagen del conjunto (test por defecto):
    cargar -> preprocesar -> [reducir] -> modelo -> [ampliar] -> postprocesar -> métricas

Conjuntos (ver src/datos.py):
    train  las 100 imágenes iniciales, en 5 folds. Para elegir hiperparámetros:
           además de la media por imagen se guarda el Dice de cada fold y su
           desviación típica entre folds (validación cruzada).
    test   el conjunto del profesor. Solo para el resultado final.

Las métricas se calculan SIEMPRE contra la máscara original, a tamaño real.
Si escala < 1, la imagen se reduce después del preprocesado, y la predicción
se amplía al tamaño original con vecino más cercano antes de postprocesar.

Se guarda en resultados/<nombre>/ (o resultados/<nombre>__train/ si se evalúa sobre train):
    metricas.csv    una fila por imagen: tiempo, TP, FP, FN, TN y todas las métricas
    resumen.json    la configuración usada, las métricas agregadas y (en train) las de cada fold
    mascaras/       las máscaras predichas (PNG 0/255), para figuras posteriores
    figuras/        original | referencia | predicción | superposición, por imagen
"""

import csv
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import cv2 as cv
import numpy as np

from src import datos, metricas, modelos, postprocesado, visualizacion
from src.preprocesado import aplicar as preprocesar
from src.preprocesado import describir as describir_pre
from src.preprocesado import redimensionar

DIR_RESULTADOS = datos.RAIZ / "resultados"
# Protocolo de evaluación actual. Los resultados sin este campo son de la
# partición antigua (70 train / 30 test) y resumen.py los oculta.
PROTOCOLO = "cv5_train+test"
CLAVES_VALIDAS = {"nombre", "descripcion", "preprocesado", "modelo", "postprocesado", "escala"}


def validar(config):
    """Comprueba la configuración antes de gastar tiempo ejecutándola."""
    desconocidas = set(config) - CLAVES_VALIDAS
    if desconocidas:
        raise ValueError(f"claves desconocidas en el experimento: {sorted(desconocidas)}")
    for clave in ("nombre", "preprocesado", "modelo"):
        if clave not in config:
            raise ValueError(f"falta la clave '{clave}' en el experimento")
    nombre = config["nombre"]
    if not nombre or any(c in nombre for c in r' /\:*?"<>|'):
        raise ValueError(f"nombre de experimento no válido para una carpeta: '{nombre}'")
    escala = config.get("escala", 1.0)
    if not 0 < escala <= 1:
        raise ValueError(f"la escala debe estar en (0, 1], no {escala}")
    modelo, _ = config["modelo"]
    if modelo not in modelos.MODELOS and modelo not in modelos.ENTRENABLES:
        raise ValueError(f"modelo desconocido '{modelo}'")


def _preparar_entrada(imagen, pasos, escala):
    entrada = preprocesar(imagen, pasos)
    return redimensionar(entrada, escala) if escala < 1 else entrada


def _normalizar_prediccion(prediccion, alto, ancho, nombre_imagen):
    """Comprueba que el modelo ha cumplido el contrato y devuelve una máscara 0/255."""
    if not isinstance(prediccion, np.ndarray) or prediccion.ndim != 2:
        raise ValueError(f"{nombre_imagen}: el modelo debe devolver una máscara 2D, "
                         f"no {getattr(prediccion, 'shape', type(prediccion))}")
    prediccion = np.where(prediccion > 0, 255, 0).astype(np.uint8)
    if prediccion.shape != (alto, ancho):
        prediccion = redimensionar(prediccion, tamano=(ancho, alto), es_mascara=True)
    return prediccion


def ejecutar(config, conjunto="test", guardar_figuras=True, verbose=True):
    """Ejecuta el experimento y devuelve el resumen de métricas."""
    validar(config)
    nombre = config["nombre"]
    pasos_pre = config["preprocesado"]
    modelo, params = config["modelo"]
    pasos_post = config.get("postprocesado", [])
    escala = config.get("escala", 1.0)

    # Los resultados de test van en resultados/<nombre>/; los de train o todo,
    # en resultados/<nombre>__train/ o __todo/, para no pisar los de test.
    carpeta = DIR_RESULTADOS / (nombre if conjunto == "test" else f"{nombre}__{conjunto}")
    if carpeta.exists():
        shutil.rmtree(carpeta)  # se rehace entero para no mezclar ejecuciones
    (carpeta / "mascaras").mkdir(parents=True)

    def log(texto):
        if verbose:
            print(texto, flush=True)

    log(f"\n=== {nombre} ({conjunto}) ===")
    log(f"  preprocesado:  {describir_pre(pasos_pre)}")
    log(f"  modelo:        {modelo} {params}")
    log(f"  postprocesado: {postprocesado.describir(pasos_post)}")
    if escala < 1:
        log(f"  escala:        {escala}")

    # Entrenamiento (solo métodos que lo necesitan). Siempre con el conjunto train.
    imagenes_train = mascaras_train = None
    tiempo_entrenamiento = 0.0
    if modelos.es_entrenable(modelo):
        if conjunto != "test":
            raise ValueError("los métodos entrenables solo se evalúan sobre test, para no evaluar con lo aprendido")
        log("  entrenando con el conjunto train...")
        imagenes_train, mascaras_train = [], []
        for n in datos.nombres("train"):
            imagen, mascara = datos.cargar_pareja(n)
            imagenes_train.append(_preparar_entrada(imagen, pasos_pre, escala))
            mascaras_train.append(redimensionar(mascara, escala, es_mascara=True) if escala < 1 else mascara)
        inicio = time.perf_counter()
    segmentar = modelos.preparar(modelo, params, imagenes_train, mascaras_train)
    if imagenes_train is not None:
        tiempo_entrenamiento = time.perf_counter() - inicio
        log(f"  entrenado en {tiempo_entrenamiento:.1f} s")

    # Evaluación imagen a imagen.
    filas = []
    lista = datos.nombres(conjunto)
    for i, n in enumerate(lista, 1):
        imagen, referencia = datos.cargar_pareja(n)
        alto, ancho = referencia.shape

        inicio = time.perf_counter()
        entrada = _preparar_entrada(imagen, pasos_pre, escala)
        prediccion = _normalizar_prediccion(segmentar(entrada), alto, ancho, n)
        prediccion = postprocesado.aplicar(prediccion, pasos_post)
        tiempo = time.perf_counter() - inicio

        fila = {"imagen": n, "tiempo_s": round(tiempo, 4)}
        if conjunto == "train":
            fila["fold"] = datos.fold_de(n)
        fila.update(metricas.calcular(prediccion, referencia))
        filas.append(fila)

        cv.imwrite(str(carpeta / "mascaras" / f"{n}.png"), prediccion)
        if guardar_figuras:
            fig = visualizacion.figura_comparacion(imagen, referencia, prediccion, f"{nombre} · {n}", fila)
            visualizacion.guardar(fig, carpeta / "figuras" / f"{n}.png", dpi=80)
        log(f"  [{i:3d}/{len(lista)}] {n}  iou {fila['iou']:.3f}  dice {fila['dice']:.3f}  ({tiempo:.2f} s)")

    # Guardar resultados.
    with open(carpeta / "metricas.csv", "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(filas[0]))
        escritor.writeheader()
        escritor.writerows(filas)

    resumen = metricas.agregar(filas)
    resumen["tiempo_medio_s"] = float(np.mean([f["tiempo_s"] for f in filas]))
    resumen["tiempo_entrenamiento_s"] = round(tiempo_entrenamiento, 2)
    if conjunto == "train":
        resumen.update(resumen_folds(filas))
    salida = {
        "nombre": nombre,
        "descripcion": config.get("descripcion", ""),
        "conjunto": conjunto,
        "protocolo": PROTOCOLO,
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "preprocesado": describir_pre(pasos_pre),
        "modelo": modelo,
        "parametros_modelo": params,
        "postprocesado": postprocesado.describir(pasos_post),
        "escala": escala,
        "config": config,
        "resumen": resumen,
    }
    with open(carpeta / "resumen.json", "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False, default=str)

    log(f"  -> IoU {resumen['media_iou']:.3f} ± {resumen['std_iou']:.3f}   "
        f"Dice {resumen['media_dice']:.3f} ± {resumen['std_dice']:.3f}   "
        f"({resumen['tiempo_medio_s']:.2f} s/imagen)")
    if conjunto == "train":
        log(f"     Dice por fold: {', '.join(f'{d:.3f}' for d in resumen['dice_por_fold'])}   "
            f"-> {resumen['media_dice_folds']:.3f} ± {resumen['std_dice_folds']:.3f} entre folds")
    log(f"  resultados en {carpeta.relative_to(datos.RAIZ)}/")
    return salida


def resumen_folds(filas, clave="dice"):
    """Media de la métrica en cada fold, y media y desviación típica entre folds.

    Validación cruzada de métodos sin entrenamiento: cada fold es un grupo de
    imágenes de validación distinto; que la métrica se parezca en todos indica
    que la configuración es estable y no depende de unas imágenes concretas.
    """
    por_fold = {}
    for fila in filas:
        por_fold.setdefault(fila["fold"], []).append(fila[clave])
    medias = [float(np.mean(por_fold[f])) for f in sorted(por_fold)]
    return {
        f"{clave}_por_fold": medias,
        f"media_{clave}_folds": float(np.mean(medias)),
        f"std_{clave}_folds": float(np.std(medias)),
    }


def cargar_resumenes(directorio=DIR_RESULTADOS):
    """Lee los resumen.json de todos los experimentos ejecutados."""
    resumenes = []
    for ruta in sorted(Path(directorio).glob("*/resumen.json")):
        with open(ruta, encoding="utf-8") as f:
            resumenes.append(json.load(f))
    return resumenes
