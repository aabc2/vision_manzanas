"""Encadena pasos de preprocesado a partir de la configuración de un experimento.

Uso:
    from src.preprocesado import aplicar

    pasos = [("gaussiano", {"k": 5}), ("clahe", {"clip": 2.0})]
    imagen_pre = aplicar(imagen, pasos)

Una lista vacía significa "sin preprocesado" y devuelve una copia de la imagen.

Para añadir un paso nuevo: escribir la función en el módulo que corresponda
(con la firma paso(imagen_bgr, **params) -> imagen_bgr) y añadirla a PASOS.
"""

from . import contraste, geometria, ruido
from ._comun import comprobar_bgr

PASOS = {
    # ruido.py
    "media": ruido.media,
    "gaussiano": ruido.gaussiano,
    "mediana": ruido.mediana,
    "bilateral": ruido.bilateral,
    # contraste.py
    "ecualizar": contraste.ecualizar,
    "clahe": contraste.clahe,
    "gamma": contraste.gamma,
    "estirar": contraste.estirar,
    # geometria.py
    "roi": geometria.roi,
}


def _normalizar(paso):
    """Acepta "clahe" o ("clahe", {...}) y devuelve siempre (nombre, params)."""
    if isinstance(paso, str):
        return paso, {}
    nombre, params = paso
    return nombre, dict(params or {})


def aplicar(imagen, pasos):
    """Aplica los pasos en orden y comprueba el contrato tras cada uno."""
    comprobar_bgr(imagen, "la imagen de entrada")
    salida = imagen.copy()
    for paso in pasos:
        nombre, params = _normalizar(paso)
        if nombre not in PASOS:
            raise ValueError(f"paso desconocido '{nombre}'. Opciones: {sorted(PASOS)}")
        try:
            salida = PASOS[nombre](salida, **params)
        except TypeError as error:
            raise TypeError(f"parámetros incorrectos en el paso '{nombre}': {error}") from error
        comprobar_bgr(salida, f"la salida del paso '{nombre}'")
        if salida.shape != imagen.shape:
            raise ValueError(f"el paso '{nombre}' ha cambiado el tamaño de la imagen")
    return salida


def describir(pasos):
    """Texto corto para tablas y nombres de archivo.

    [("gaussiano", {"k": 5}), ("clahe", {"clip": 2.0})] -> "gaussiano(k=5) + clahe(clip=2.0)"
    """
    if not pasos:
        return "sin preprocesado"
    partes = []
    for paso in pasos:
        nombre, params = _normalizar(paso)
        args = ", ".join(f"{k}={v}" for k, v in params.items())
        partes.append(f"{nombre}({args})" if args else nombre)
    return " + ".join(partes)
