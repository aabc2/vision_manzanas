"""Corrección de iluminación y contraste (Tema 3: transformaciones de intensidad;
Tema 6: histograma, ecualización y CLAHE; Práctica 2).

IMPORTANTE: todos estos pasos modifican SOLO la luminancia (canal L de Lab) y
dejan intactos los canales de color a y b. Si se ecualizaran B, G y R por
separado, cada canal se estiraría de forma distinta y cambiaría el tono de la
imagen, que es justo la información que necesitamos para encontrar las
manzanas.
"""

import cv2 as cv
import numpy as np

from ._comun import comprobar_bgr


def _sobre_luminancia(imagen, transformar):
    """Pasa la imagen a Lab, aplica `transformar` al canal L y vuelve a BGR."""
    comprobar_bgr(imagen)
    lab = cv.cvtColor(imagen, cv.COLOR_BGR2Lab)
    l, a, b = cv.split(lab)
    l = transformar(l)
    return cv.cvtColor(cv.merge((l, a, b)), cv.COLOR_Lab2BGR)


def ecualizar(imagen):
    """Ecualización global del histograma de la luminancia.

    Reparte los niveles de brillo por todo el rango [0, 255]. Al ser global,
    puede sobreexponer el cielo o amplificar el ruido en las sombras.
    """
    return _sobre_luminancia(imagen, cv.equalizeHist)


def clahe(imagen, clip=2.0, rejilla=8):
    """Ecualización adaptativa con contraste limitado (CLAHE).

    Divide la imagen en una rejilla de `rejilla` x `rejilla` regiones y
    ecualiza cada una por separado, así que corrige mejor la iluminación
    desigual (sol directo y sombras) que la ecualización global.

    clip: límite de contraste. Valores altos realzan más pero amplifican el
          ruido; valores cercanos a 1 apenas modifican la imagen.
    """
    operador = cv.createCLAHE(clipLimit=clip, tileGridSize=(rejilla, rejilla))
    return _sobre_luminancia(imagen, operador.apply)


def gamma(imagen, valor=1.0):
    """Ley de potencias s = 255 * (r / 255) ** valor sobre la luminancia.

    valor < 1 aclara las sombras; valor > 1 oscurece los tonos claros.
    Se precalcula una tabla de 256 valores y se aplica con cv.LUT.
    """
    if valor <= 0:
        raise ValueError(f"gamma debe ser positivo, no {valor}")
    tabla = (255.0 * (np.arange(256) / 255.0) ** valor).round().astype(np.uint8)
    return _sobre_luminancia(imagen, lambda l: cv.LUT(l, tabla))


def estirar(imagen, p_bajo=1.0, p_alto=99.0):
    """Estiramiento lineal del contraste de la luminancia.

    Los percentiles p_bajo y p_alto de L pasan a 0 y 255, y los valores entre
    medias se reparten linealmente (transformación lineal por tramos). Usar
    percentiles en vez del mínimo y el máximo evita que unos pocos píxeles
    del cielo o de una sombra muy oscura anulen el efecto.
    """
    if not 0 <= p_bajo < p_alto <= 100:
        raise ValueError("se necesita 0 <= p_bajo < p_alto <= 100")

    def _estirar(l):
        r1, r2 = np.percentile(l, (p_bajo, p_alto))
        if r2 <= r1:
            return l
        salida = (l.astype(np.float32) - r1) * 255.0 / (r2 - r1)
        return np.clip(salida, 0, 255).astype(np.uint8)

    return _sobre_luminancia(imagen, _estirar)
