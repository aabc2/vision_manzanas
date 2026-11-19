"""Eliminación de ruido (Tema 3: filtros y convolución; Práctica 2).

Todos los filtros trabajan sobre los tres canales BGR a la vez, por lo que
conservan el color de la imagen.

Cuándo usar cada uno:
    - media:     el más simple; difumina por igual bordes y ruido.
    - gaussiano: pondera más los píxeles cercanos; difumina menos que la media
                 con el mismo tamaño. Es el habitual antes de umbralizar.
    - mediana:   elimina píxeles aislados extremos (ruido sal y pimienta) y
                 conserva mejor los bordes que la media.
    - bilateral: suaviza zonas homogéneas y conserva los bordes, porque solo
                 promedia píxeles cercanos Y de color parecido. Es más lento.
"""

import cv2 as cv

from ._comun import comprobar_bgr, comprobar_impar


def media(imagen, k=5):
    """Filtro de media con un kernel k x k."""
    comprobar_bgr(imagen)
    comprobar_impar(k, "k")
    return cv.blur(imagen, (k, k))


def gaussiano(imagen, k=5, sigma=0):
    """Filtro gaussiano k x k. Con sigma=0, OpenCV lo calcula a partir de k."""
    comprobar_bgr(imagen)
    comprobar_impar(k, "k")
    return cv.GaussianBlur(imagen, (k, k), sigma)


def mediana(imagen, k=5):
    """Filtro de mediana k x k."""
    comprobar_bgr(imagen)
    comprobar_impar(k, "k")
    return cv.medianBlur(imagen, k)


def bilateral(imagen, d=9, sigma_color=75, sigma_espacio=75):
    """Filtro bilateral.

    d:             diámetro de la vecindad en píxeles.
    sigma_color:   cuánto pueden diferir dos colores para promediarse
                   (mayor valor = mezcla colores más distintos).
    sigma_espacio: cuánto influyen los píxeles lejanos.
    """
    comprobar_bgr(imagen)
    return cv.bilateralFilter(imagen, d, sigma_color, sigma_espacio)
