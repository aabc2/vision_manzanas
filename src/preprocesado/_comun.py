"""Comprobaciones compartidas por todos los pasos de preprocesado.

Contrato del preprocesado (lo usan todos los métodos del proyecto):
    - Entrada: imagen BGR, uint8, forma (alto, ancho, 3), tal como la devuelve cv.imread.
    - Salida: imagen BGR, uint8, con la MISMA forma que la entrada.

Así cualquier paso se puede encadenar con cualquier otro, y la máscara de
referencia sigue siendo válida para evaluar el resultado.
"""

import numpy as np


def comprobar_bgr(imagen, nombre="imagen"):
    """Lanza un error claro si la imagen no cumple el contrato BGR uint8."""
    if not isinstance(imagen, np.ndarray):
        raise TypeError(f"{nombre} debe ser un array de NumPy, no {type(imagen).__name__}")
    if imagen.dtype != np.uint8:
        raise TypeError(f"{nombre} debe ser uint8, no {imagen.dtype}")
    if imagen.ndim != 3 or imagen.shape[2] != 3:
        raise ValueError(f"{nombre} debe tener forma (alto, ancho, 3), no {imagen.shape}")


def comprobar_impar(valor, nombre):
    """Los tamaños de kernel de OpenCV tienen que ser impares y positivos."""
    if valor < 1 or valor % 2 == 0:
        raise ValueError(f"{nombre} debe ser un entero impar positivo, no {valor}")
