"""Espacios de color y extracción de canales (Tema 2; Práctica 1).

Estas funciones NO forman parte del pipeline, porque devuelven imágenes que
ya no son BGR. Las usan los modelos para elegir en qué espacio trabajar:

    from src.preprocesado.color import canal, convertir
    a = canal(imagen, "a")          # canal a* de Lab (verde <-> rojo)
    hsv = convertir(imagen, "hsv")  # los tres canales H, S, V

Rangos de OpenCV para imágenes uint8 (distintos de los de teoría):
    - H de HSV va de 0 a 179 (grados / 2). El rojo está en los dos extremos.
    - L, a y b de Lab van de 0 a 255; a = b = 128 es el gris neutro.
    - Cr y Cb van de 0 a 255; 128 es el gris neutro.

Lo que se ha visto en el dataset (manzana frente a fondo):
    - Manzanas rojas: a* y Cr mucho más altos que el fondo.
    - Manzanas amarillas: b* más alto y H muy concentrado (~31) frente a las
      hojas (~45). En a* NO se distinguen del fondo.
"""

import cv2 as cv

from ._comun import comprobar_bgr

# Conversión desde BGR y nombre de cada canal del espacio resultante.
ESPACIOS = {
    "bgr": (None, ("B", "G", "R")),
    "hsv": (cv.COLOR_BGR2HSV, ("H", "S", "V")),
    "lab": (cv.COLOR_BGR2Lab, ("L", "a", "b")),
    "ycrcb": (cv.COLOR_BGR2YCrCb, ("Y", "Cr", "Cb")),
}

# Canal -> (espacio, posición dentro del espacio).
CANALES = {
    nombre: (espacio, i)
    for espacio, (_, nombres) in ESPACIOS.items()
    for i, nombre in enumerate(nombres)
}


def convertir(imagen, espacio):
    """Convierte una imagen BGR al espacio indicado: bgr, hsv, lab o ycrcb."""
    comprobar_bgr(imagen)
    espacio = espacio.lower()
    if espacio not in ESPACIOS:
        raise ValueError(f"espacio desconocido '{espacio}'. Opciones: {list(ESPACIOS)}")
    codigo, _ = ESPACIOS[espacio]
    return imagen.copy() if codigo is None else cv.cvtColor(imagen, codigo)


def a_gris(imagen):
    """Escala de grises (luminancia ponderada de OpenCV)."""
    comprobar_bgr(imagen)
    return cv.cvtColor(imagen, cv.COLOR_BGR2GRAY)


def canal(imagen, nombre):
    """Devuelve un único canal como imagen 2D uint8.

    nombre: "gris", "B", "G", "R", "H", "S", "V", "L", "a", "b", "Y", "Cr" o "Cb".
    """
    if nombre == "gris":
        return a_gris(imagen)
    if nombre not in CANALES:
        opciones = ["gris"] + list(CANALES)
        raise ValueError(f"canal desconocido '{nombre}'. Opciones: {opciones}")
    espacio, i = CANALES[nombre]
    return convertir(imagen, espacio)[:, :, i]
