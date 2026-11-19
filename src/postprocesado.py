"""Postprocesado de la máscara predicha (Tema 3: morfología; Tema 5: componentes conectadas).

Base mínima para que el pipeline funcione. La persona B lo amplía (por ejemplo,
con la separación de manzanas pegadas mediante transformada de distancia y
watershed) manteniendo el mismo contrato:

    paso(mascara, **params) -> mascara
    con mascara uint8 de valores 0 / 255, y el mismo tamaño a la entrada y a la salida.

Uso:
    from src.postprocesado import aplicar
    limpia = aplicar(mascara, [("apertura", {"k": 5}), ("filtrar_area", {"minima": 100})])
"""

import cv2 as cv
import numpy as np


def _comprobar(mascara, nombre="la máscara"):
    if mascara.dtype != np.uint8 or mascara.ndim != 2:
        raise ValueError(f"{nombre} debe ser uint8 de 2 dimensiones, no {mascara.dtype} {mascara.shape}")


def _elemento(k, forma):
    formas = {"elipse": cv.MORPH_ELLIPSE, "rect": cv.MORPH_RECT, "cruz": cv.MORPH_CROSS}
    if forma not in formas:
        raise ValueError(f"forma desconocida '{forma}'. Opciones: {list(formas)}")
    if k < 1 or k % 2 == 0:
        raise ValueError(f"k debe ser impar y positivo, no {k}")
    return cv.getStructuringElement(formas[forma], (k, k))


def apertura(mascara, k=5, forma="elipse"):
    """Erosión + dilatación: elimina manchas pequeñas sin encoger lo que queda."""
    return cv.morphologyEx(mascara, cv.MORPH_OPEN, _elemento(k, forma))


def cierre(mascara, k=5, forma="elipse"):
    """Dilatación + erosión: rellena huecos pequeños y une trozos muy cercanos."""
    return cv.morphologyEx(mascara, cv.MORPH_CLOSE, _elemento(k, forma))


def filtrar_area(mascara, minima=0, maxima=None):
    """Elimina las componentes conectadas con área fuera de [minima, maxima] píxeles."""
    n, etiquetas, stats, _ = cv.connectedComponentsWithStats(mascara, connectivity=8)
    areas = stats[:, cv.CC_STAT_AREA]
    validas = areas >= minima
    if maxima is not None:
        validas &= areas <= maxima
    validas[0] = False  # la etiqueta 0 es el fondo
    return np.where(validas[etiquetas], 255, 0).astype(np.uint8)


PASOS = {
    "apertura": apertura,
    "cierre": cierre,
    "filtrar_area": filtrar_area,
}


def aplicar(mascara, pasos):
    """Aplica los pasos en orden y comprueba el contrato tras cada uno."""
    _comprobar(mascara)
    salida = np.where(mascara > 0, 255, 0).astype(np.uint8)
    for paso in pasos:
        nombre, params = (paso, {}) if isinstance(paso, str) else (paso[0], dict(paso[1] or {}))
        if nombre not in PASOS:
            raise ValueError(f"paso de postprocesado desconocido '{nombre}'. Opciones: {sorted(PASOS)}")
        try:
            salida = PASOS[nombre](salida, **params)
        except TypeError as error:
            raise TypeError(f"parámetros incorrectos en el paso '{nombre}': {error}") from error
        _comprobar(salida, f"la salida del paso '{nombre}'")
        if salida.shape != mascara.shape:
            raise ValueError(f"el paso '{nombre}' ha cambiado el tamaño de la máscara")
    return salida


def describir(pasos):
    if not pasos:
        return "sin postprocesado"
    partes = []
    for paso in pasos:
        nombre, params = (paso, {}) if isinstance(paso, str) else (paso[0], dict(paso[1] or {}))
        args = ", ".join(f"{k}={v}" for k, v in params.items())
        partes.append(f"{nombre}({args})" if args else nombre)
    return " + ".join(partes)
