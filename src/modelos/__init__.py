"""Registro de los métodos de segmentación.

Hay dos tipos de método:

1. Sin entrenamiento (umbralización, clustering, regiones...). Una función:
       metodo(imagen_bgr, **params) -> máscara uint8 0/255 del mismo alto y ancho
   Se registra en MODELOS.

2. Con entrenamiento. Una función que aprende de las imágenes de train y
   devuelve un segmentador ya listo:
       entrenar(imagenes, mascaras, **params) -> segmentador
       segmentador(imagen_bgr) -> máscara uint8 0/255
   Se registra en ENTRENABLES. `imagenes` y `mascaras` son listas ya
   preprocesadas (y reducidas, si el experimento usa escala < 1).

Un archivo por algoritmo, y cada uno añade aquí solo sus líneas, así no hay
conflictos en Git:
    Persona 1: umbral_global.py, watershed.py
    Persona 2: otsu.py, regiones.py
    Persona 3: adaptativa.py, kmeans.py
"""

from . import otsu

MODELOS = {
    # Persona 1: umbralización global y watershed
    # "umbral_global": umbral_global.umbral_global,
    # Persona 2: Otsu y crecimiento de regiones
    "otsu": otsu.otsu,
    "otsu_combinado": otsu.otsu_combinado,
    # Persona 3: umbralización adaptativa y K-means
    # "adaptativa": adaptativa.adaptativa,
}

ENTRENABLES = {
}


def preparar(nombre, params=None, imagenes_train=None, mascaras_train=None):
    """Devuelve una función imagen_bgr -> máscara lista para usar.

    Para los métodos entrenables, entrena primero con las imágenes de train.
    """
    params = dict(params or {})
    if nombre in MODELOS:
        funcion = MODELOS[nombre]
        return lambda imagen: funcion(imagen, **params)
    if nombre in ENTRENABLES:
        if not imagenes_train:
            raise ValueError(f"el método '{nombre}' necesita imágenes de entrenamiento")
        return ENTRENABLES[nombre](imagenes_train, mascaras_train, **params)
    raise ValueError(f"método desconocido '{nombre}'. Opciones: {sorted(MODELOS) + sorted(ENTRENABLES)}")


def es_entrenable(nombre):
    return nombre in ENTRENABLES
