"""Carga del dataset y partición fija en entrenamiento y test.

Estructura esperada (no se sube a GitHub, cada uno la tiene en local):
    data/imagenes/img_XXX.png   imagen BGR de 720 x 1280
    data/mascaras/img_XXX.png   máscara de referencia, 0 = fondo y 255 = manzana

La partición está en src/particion.json y SÍ se sube a GitHub, para que los
tres evaluemos exactamente sobre las mismas imágenes. Se crea una sola vez con
scripts/crear_particion.py.

Uso:
    from src.datos import cargar_pareja, nombres

    for nombre in nombres("test"):
        imagen, mascara = cargar_pareja(nombre)
"""

import json
from pathlib import Path

import cv2 as cv
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
DIR_IMAGENES = RAIZ / "data" / "imagenes"
DIR_MASCARAS = RAIZ / "data" / "mascaras"
RUTA_PARTICION = Path(__file__).resolve().parent / "particion.json"

CONJUNTOS = ("train", "test", "todo")


def listar_disponibles():
    """Nombres (sin extensión) de las imágenes que tienen máscara, ordenados."""
    imagenes = {p.stem for p in DIR_IMAGENES.glob("*.png")}
    mascaras = {p.stem for p in DIR_MASCARAS.glob("*.png")}
    sin_pareja = imagenes ^ mascaras
    if sin_pareja:
        raise FileNotFoundError(f"imágenes o máscaras sin pareja: {sorted(sin_pareja)}")
    if not imagenes:
        raise FileNotFoundError(f"no hay imágenes en {DIR_IMAGENES}. ¿Has descomprimido el dataset en data/?")
    return sorted(imagenes)


def cargar_imagen(nombre):
    """Imagen BGR uint8, tal como la devuelve cv.imread."""
    ruta = DIR_IMAGENES / f"{nombre}.png"
    imagen = cv.imread(str(ruta), cv.IMREAD_COLOR)
    if imagen is None:
        raise FileNotFoundError(f"no se pudo leer {ruta}")
    return imagen


def cargar_mascara(nombre):
    """Máscara de referencia uint8 con valores 0 (fondo) y 255 (manzana)."""
    ruta = DIR_MASCARAS / f"{nombre}.png"
    mascara = cv.imread(str(ruta), cv.IMREAD_GRAYSCALE)
    if mascara is None:
        raise FileNotFoundError(f"no se pudo leer {ruta}")
    # Por si algún editor ha dejado grises en los bordes al guardar.
    return np.where(mascara > 127, 255, 0).astype(np.uint8)


def cargar_pareja(nombre):
    """(imagen BGR, máscara 0/255) de una imagen del dataset."""
    imagen = cargar_imagen(nombre)
    mascara = cargar_mascara(nombre)
    if imagen.shape[:2] != mascara.shape:
        raise ValueError(f"{nombre}: la imagen {imagen.shape[:2]} y la máscara {mascara.shape} no miden lo mismo")
    return imagen, mascara


def leer_particion():
    """Diccionario {"train": [...], "test": [...]} leído de src/particion.json."""
    if not RUTA_PARTICION.exists():
        raise FileNotFoundError(
            f"no existe {RUTA_PARTICION.name}. Créala una sola vez con: "
            "uv run python scripts/crear_particion.py"
        )
    with open(RUTA_PARTICION, encoding="utf-8") as f:
        return json.load(f)


def nombres(conjunto="test"):
    """Nombres de las imágenes de un conjunto: "train", "test" o "todo"."""
    if conjunto not in CONJUNTOS:
        raise ValueError(f"conjunto desconocido '{conjunto}'. Opciones: {CONJUNTOS}")
    if conjunto == "todo":
        return listar_disponibles()

    lista = leer_particion()[conjunto]
    faltan = sorted(set(lista) - set(listar_disponibles()))
    if faltan:
        raise FileNotFoundError(f"la partición incluye imágenes que no tienes en data/: {faltan[:5]}...")
    return lista
