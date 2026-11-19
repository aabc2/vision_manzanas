"""Operaciones geométricas (Tema 3: escalado, interpolación y recorte).

- roi() sí forma parte del pipeline: anula lo que queda fuera de la región de
  interés, pero mantiene el tamaño de la imagen.
- redimensionar() NO forma parte del pipeline, porque cambia el tamaño. La
  usa la evaluación para acelerar los modelos lentos: se reduce la imagen, se
  segmenta, y la máscara predicha se devuelve al tamaño original antes de
  calcular las métricas.
"""

import cv2 as cv

from ._comun import comprobar_bgr


def roi(imagen, y_ini=0.0, y_fin=1.0, x_ini=0.0, x_fin=1.0):
    """Pone a negro todo lo que queda fuera del rectángulo indicado.

    Los límites son fracciones de la imagen (0 = arriba/izquierda,
    1 = abajo/derecha), así que valen para cualquier resolución.
    Ejemplo: roi(img, y_fin=0.75) descarta el 25 % inferior.

    Un píxel negro (0, 0, 0) no tiene color de manzana, así que los métodos
    basados en color lo clasificarán como fondo.
    """
    comprobar_bgr(imagen)
    if not (0 <= y_ini < y_fin <= 1 and 0 <= x_ini < x_fin <= 1):
        raise ValueError("los límites deben cumplir 0 <= ini < fin <= 1")
    alto, ancho = imagen.shape[:2]
    y1, y2 = round(y_ini * alto), round(y_fin * alto)
    x1, x2 = round(x_ini * ancho), round(x_fin * ancho)

    salida = imagen.copy()
    salida[:y1] = 0
    salida[y2:] = 0
    salida[:, :x1] = 0
    salida[:, x2:] = 0
    return salida


def redimensionar(imagen, escala=None, tamano=None, es_mascara=False):
    """Cambia el tamaño de una imagen o de una máscara.

    escala: factor (0.5 = la mitad de alto y de ancho).
    tamano: (ancho, alto) de destino, como en cv.resize. Tiene prioridad.
    es_mascara: si es True se usa interpolación al vecino más cercano, para
        que la máscara siga teniendo solo los valores 0 y 255. Con cualquier
        otra interpolación aparecerían grises en los bordes.
    """
    if tamano is None:
        if escala is None or escala <= 0:
            raise ValueError("indica una escala positiva o un tamano (ancho, alto)")
        alto, ancho = imagen.shape[:2]
        tamano = (max(1, round(ancho * escala)), max(1, round(alto * escala)))

    if es_mascara:
        interpolacion = cv.INTER_NEAREST
    elif tamano[0] < imagen.shape[1]:
        interpolacion = cv.INTER_AREA  # la recomendada por OpenCV para reducir
    else:
        interpolacion = cv.INTER_LINEAR
    return cv.resize(imagen, tamano, interpolation=interpolacion)
