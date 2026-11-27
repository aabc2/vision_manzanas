"""Carga del dataset: conjunto de desarrollo (train, con 5 folds) y test del profesor.

Estructura esperada (no se sube a GitHub, cada uno la tiene en local):
    data/train/imagenes/img_000.png ... img_099.png   las 100 imágenes iniciales
    data/train/mascaras/img_000.png ... img_099.png
    data/test/imagenes/img_150.png ... img_179.png    el test que dio el profesor
    data/test/mascaras/img_150.png ... img_179.png
Las máscaras valen 0 (fondo) y 255 (manzana).

Protocolo (lo que pidió el profesor):
    - train: las 100 imágenes iniciales, repartidas en K = 5 folds fijos.
      Los hiperparámetros se eligen con validación cruzada: el Dice de cada
      combinación se calcula en cada fold y se elige la de mejor media.
    - test: SOLO para el resultado final de la configuración ya elegida.
      Nunca se mezcla con train ni se ajusta nada mirándolo.

Los folds están en src/particion.json y SÍ se suben a GitHub, para que los tres
usemos exactamente los mismos. Se crean una sola vez con scripts/crear_particion.py.

Uso:
    from src.datos import cargar_pareja, nombres, fold_de

    for nombre in nombres("train"):          # o "test"
        imagen, mascara = cargar_pareja(nombre)
        fold = fold_de(nombre)               # 0..4 (solo imágenes de train)
"""

import json
from functools import lru_cache
from pathlib import Path

import cv2 as cv
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
DIR_DATOS = RAIZ / "data"
DIRECTORIOS = {
    "train": DIR_DATOS / "train",
    "test": DIR_DATOS / "test",
}
# Compatibilidad con código anterior: las rutas "por defecto" son las de train.
DIR_IMAGENES = DIRECTORIOS["train"] / "imagenes"
DIR_MASCARAS = DIRECTORIOS["train"] / "mascaras"
RUTA_PARTICION = Path(__file__).resolve().parent / "particion.json"

CONJUNTOS = ("train", "test")


def _comprobar_estructura():
    """Error claro si alguien tiene todavía la estructura antigua (data/imagenes)."""
    if (DIR_DATOS / "imagenes").exists() and not DIRECTORIOS["train"].exists():
        raise FileNotFoundError(
            "La estructura de data/ ha cambiado. Mueve data/imagenes y data/mascaras a "
            "data/train/imagenes y data/train/mascaras, y pon el test del profesor en "
            "data/test/imagenes y data/test/mascaras (ver README)."
        )


def listar_disponibles(conjunto="train"):
    """Nombres (sin extensión) de las imágenes de un conjunto que tienen máscara, ordenados."""
    if conjunto not in CONJUNTOS:
        raise ValueError(f"conjunto desconocido '{conjunto}'. Opciones: {CONJUNTOS}")
    _comprobar_estructura()
    carpeta = DIRECTORIOS[conjunto]
    imagenes = {p.stem for p in (carpeta / "imagenes").glob("*.png")}
    mascaras = {p.stem for p in (carpeta / "mascaras").glob("*.png")}
    sin_pareja = imagenes ^ mascaras
    if sin_pareja:
        raise FileNotFoundError(f"{conjunto}: imágenes o máscaras sin pareja: {sorted(sin_pareja)}")
    if not imagenes:
        raise FileNotFoundError(f"no hay imágenes en {carpeta / 'imagenes'}. ¿Has colocado el dataset? (ver README)")
    return sorted(imagenes)


@lru_cache(maxsize=None)
def _ubicacion(nombre):
    """Carpeta (train o test) donde está una imagen. Los nombres no se repiten entre conjuntos."""
    _comprobar_estructura()
    encontradas = [c for c, d in DIRECTORIOS.items() if (d / "imagenes" / f"{nombre}.png").exists()]
    if not encontradas:
        raise FileNotFoundError(f"no existe {nombre}.png ni en data/train ni en data/test")
    if len(encontradas) > 1:
        raise ValueError(f"{nombre}.png está a la vez en train y en test: los conjuntos no se pueden mezclar")
    return DIRECTORIOS[encontradas[0]]


def cargar_imagen(nombre):
    """Imagen BGR uint8, tal como la devuelve cv.imread."""
    ruta = _ubicacion(nombre) / "imagenes" / f"{nombre}.png"
    imagen = cv.imread(str(ruta), cv.IMREAD_COLOR)
    if imagen is None:
        raise FileNotFoundError(f"no se pudo leer {ruta}")
    return imagen


def cargar_mascara(nombre):
    """Máscara de referencia uint8 con valores 0 (fondo) y 255 (manzana)."""
    ruta = _ubicacion(nombre) / "mascaras" / f"{nombre}.png"
    mascara = cv.imread(str(ruta), cv.IMREAD_GRAYSCALE)
    if mascara is None:
        raise FileNotFoundError(f"no se pudo leer {ruta}")
    # Por si algún editor ha dejado grises en los bordes al guardar.
    return np.where(mascara > 127, 255, 0).astype(np.uint8)


def cargar_pareja(nombre):
    """(imagen BGR, máscara 0/255) de una imagen del dataset (de train o de test)."""
    imagen = cargar_imagen(nombre)
    mascara = cargar_mascara(nombre)
    if imagen.shape[:2] != mascara.shape:
        raise ValueError(f"{nombre}: la imagen {imagen.shape[:2]} y la máscara {mascara.shape} no miden lo mismo")
    return imagen, mascara


def leer_particion():
    """Diccionario {"k": 5, "semilla": ..., "folds": [[...], ...]} leído de src/particion.json."""
    if not RUTA_PARTICION.exists():
        raise FileNotFoundError(
            f"no existe {RUTA_PARTICION.name}. Créala una sola vez con: "
            "uv run python scripts/crear_particion.py"
        )
    with open(RUTA_PARTICION, encoding="utf-8") as f:
        particion = json.load(f)
    if "folds" not in particion:
        raise ValueError(
            f"{RUTA_PARTICION.name} tiene el formato antiguo (70/30). Haz git pull para tener "
            "la versión con folds, o créala con: uv run python scripts/crear_particion.py --forzar"
        )
    return particion


def folds():
    """Lista con las K listas de nombres de train, una por fold (siempre las mismas)."""
    particion = leer_particion()
    disponibles = set(listar_disponibles("train"))
    faltan = sorted({n for f in particion["folds"] for n in f} - disponibles)
    if faltan:
        raise FileNotFoundError(f"la partición incluye imágenes que no tienes en data/train: {faltan[:5]}...")
    return [list(f) for f in particion["folds"]]


@lru_cache(maxsize=None)
def _mapa_folds():
    return {n: i for i, lista in enumerate(leer_particion()["folds"]) for n in lista}


def fold_de(nombre):
    """Índice del fold (0..K-1) al que pertenece una imagen de train; None si es de test."""
    return _mapa_folds().get(nombre)


def nombres(conjunto="test"):
    """Nombres de las imágenes de un conjunto: "train" (las de los folds) o "test" (el del profesor)."""
    if conjunto not in CONJUNTOS:
        raise ValueError(f"conjunto desconocido '{conjunto}'. Opciones: {CONJUNTOS}")
    if conjunto == "test":
        return listar_disponibles("test")
    return sorted(n for f in folds() for n in f)
