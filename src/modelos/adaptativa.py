"""Umbralización adaptativa (Tema 5) - Persona 3.

Firma común: metodo(imagen_bgr, **params) -> máscara uint8 (0 fondo, 255 manzana).

Idea (Tema 5, "umbralización adaptativa" y "umbralización local"):
    En vez de un único umbral para toda la imagen, cada píxel tiene su propio
    umbral T(x, y), calculado con los píxeles de su vecindad (un cuadrado de
    `bloque` x `bloque` píxeles centrado en él). Un píxel es manzana si
    destaca sobre su vecindad.

Métodos:
    - "media":     T = media de la vecindad - C
    - "gaussiana": T = suma ponderada Gaussiana de la vecindad - C
                   (los vecinos cercanos pesan más que los lejanos)
      Los dos con cv.adaptiveThreshold.
    - "niblack":   T = media + k * desviación típica - C
      Implementado con cv.boxFilter (media de la ventana), porque
      cv.ximgproc.niBlackThreshold solo está en opencv-contrib, que no usamos.
      La desviación típica sale de: var = media(x^2) - media(x)^2.

Por qué puede ayudar en este dataset (análisis de errores de Otsu):
    Las manzanas en sombra tienen Cr y b* más bajos que las iluminadas, y un
    umbral global no sirve para la luz y la sombra a la vez. Un umbral local
    compara cada manzana con lo que la rodea (sus hojas, en su misma luz), así
    que una manzana en sombra también destaca en su zona.

Cómo elegir los parámetros:
    - C NEGATIVO: con T = media - C y C < 0, el píxel tiene que superar la media
      de su vecindad en |C| niveles. Con C >= 0, en zonas uniformes (cielo,
      hojas) la mitad de los píxeles quedaría por encima de su media y saldría
      mucho fondo marcado.
    - `bloque` MAYOR que una manzana: si la ventana cabe dentro de la manzana,
      el centro de la manzana solo se compara consigo mismo y queda hueco.
      Tiene que ser impar (lo exige OpenCV).
"""

import cv2 as cv
import numpy as np

from src.preprocesado import canal as extraer_canal

METODOS = ("media", "gaussiana", "niblack")
_OPENCV = {"media": cv.ADAPTIVE_THRESH_MEAN_C, "gaussiana": cv.ADAPTIVE_THRESH_GAUSSIAN_C}


def _comprobar_bloque(bloque):
    if bloque < 3 or bloque % 2 == 0:
        raise ValueError(f"bloque debe ser impar y >= 3, no {bloque}")


def _niblack(valores, bloque, k, C):
    """Máscara de Niblack: píxel > media + k * desviación - C."""
    x = valores.astype(np.float32)
    media = cv.boxFilter(x, -1, (bloque, bloque), borderType=cv.BORDER_REFLECT)
    media_cuadrados = cv.boxFilter(x * x, -1, (bloque, bloque), borderType=cv.BORDER_REFLECT)
    desviacion = np.sqrt(np.maximum(media_cuadrados - media * media, 0))
    umbral = media + k * desviacion - C
    return np.where(x > umbral, 255, 0).astype(np.uint8)


def _segmentar_canal(imagen, canal, metodo, bloque, C, k, invertir):
    _comprobar_bloque(bloque)
    valores = extraer_canal(imagen, canal)
    if invertir:
        valores = 255 - valores  # la manzana pasa a ser lo que destaca por arriba
    if metodo in _OPENCV:
        # THRESH_BINARY: 255 si valor > T(x, y), con T = media (o Gaussiana) - C.
        return cv.adaptiveThreshold(valores, 255, _OPENCV[metodo], cv.THRESH_BINARY, bloque, C)
    if metodo == "niblack":
        return _niblack(valores, bloque, k, C)
    raise ValueError(f"método desconocido '{metodo}'. Opciones: {METODOS}")


def adaptativa(imagen, canal="Cr", metodo="media", bloque=101, C=-10, k=0.5, invertir=False):
    """Umbralización adaptativa sobre un único canal.

    canal:    "gris", "Cr", "a", "b", "H", "S", "V"... (ver src.preprocesado.color)
    metodo:   "media", "gaussiana" o "niblack"
    bloque:   lado de la vecindad en píxeles (impar)
    C:        constante que se resta al umbral (negativa = más exigente)
    k:        peso de la desviación típica (solo niblack)
    invertir: True si la manzana es MÁS OSCURA que el fondo en ese canal (p. ej. H)
    """
    return _segmentar_canal(imagen, canal, metodo, bloque, C, k, invertir)


def adaptativa_combinada(imagen, canales=("Cr", "b"), operacion="and", metodo="media",
                         bloque=101, C=-10, k=0.5, invertir=()):
    """Umbralización adaptativa en varios canales y combinación de las máscaras.

    Mismo motivo que en Otsu: Cr separa las manzanas rojas y b* las amarillas.
    "and" se queda con lo que marcan los dos canales (más precisión);
    "or" con lo que marque cualquiera (más recall).
    invertir: canales a invertir, por ejemplo ("H",).
    """
    if operacion not in ("and", "or"):
        raise ValueError("operacion debe ser 'and' u 'or'")
    mascaras = [_segmentar_canal(imagen, c, metodo, bloque, C, k, c in invertir) for c in canales]
    resultado = mascaras[0]
    for m in mascaras[1:]:
        resultado = np.bitwise_and(resultado, m) if operacion == "and" else np.bitwise_or(resultado, m)
    return resultado