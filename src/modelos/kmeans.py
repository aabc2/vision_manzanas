"""K-means (Tema 5, segmentación basada en agrupaciones) - Persona 3.

Firma común: metodo(imagen_bgr, **params) -> máscara uint8 (0 fondo, 255 manzana).

Algoritmo (cv.kmeans, imagen a imagen, sin entrenamiento):
    1. Cada píxel se convierte en un vector de características de color
       (por ejemplo, sus valores de Cr y b*).
    2. K-means agrupa los vectores en K clústeres:
         - inicialización aleatoria de los K centroides (cv.KMEANS_RANDOM_CENTERS,
           como en la teoría);
         - cada píxel se asigna al centroide más cercano;
         - cada centroide se recalcula como la media de sus píxeles;
         - se repite hasta que los centroides casi no se mueven o se llega
           al máximo de iteraciones.
       Como K-means es sensible a la inicialización, se repite `intentos`
       veces y OpenCV se queda con la solución más compacta.
    3. K-means no sabe qué clúster es la manzana: solo agrupa colores. Se elige
       con una regla sobre el color MEDIO de cada clúster en Cr y b* (ver
       `seleccion`), porque Cr separa las manzanas rojas y b* las amarillas.

Para que sea rápido, los centroides se calculan con una muestra aleatoria de
`muestras` píxeles, y después TODOS los píxeles se asignan a su centroide más
cercano (que es el paso de asignación del propio K-means). Con una semilla fija
el resultado es reproducible.

Limitación de la teoría que hay que tener en cuenta: K-means no usa la posición
de los píxeles (no refuerza la continuidad espacial), y la manzana ocupa ~5 %
de la imagen. Con K pequeño, la manzana acaba mezclada en un clúster con algo
del fondo; por eso se prueban varios K.
"""

import cv2 as cv
import numpy as np

from src.preprocesado import canal as extraer_canal

# Canales que forman el vector de características de cada píxel.
CARACTERISTICAS = {
    "bgr": ("B", "G", "R"),     # sin cambio de espacio de color (línea base)
    "lab": ("L", "a", "b"),
    "ab": ("a", "b"),           # Lab sin la luminosidad
    "Cr_b": ("Cr", "b"),        # los dos canales que mejor funcionan en Otsu
}
SELECCIONES = ("max_Cr", "max_suma", "max_Cr_o_max_b", "relativa")


def _vectores(imagen, caracteristicas):
    """Matriz (número de píxeles, d) en float32, como la pide cv.kmeans."""
    if caracteristicas not in CARACTERISTICAS:
        raise ValueError(f"características desconocidas '{caracteristicas}'. Opciones: {list(CARACTERISTICAS)}")
    canales = [extraer_canal(imagen, c).reshape(-1) for c in CARACTERISTICAS[caracteristicas]]
    return np.stack(canales, axis=1).astype(np.float32)


def _asignar(vectores, centros):
    """Índice del centroide más cercano para cada píxel (distancia euclídea).

    Usa |x - c|^2 = |x|^2 - 2 x·c + |c|^2 para no crear una matriz enorme.
    |x|^2 no depende del centroide, así que no hace falta para el argmin.
    """
    distancias = -2.0 * vectores @ centros.T + (centros ** 2).sum(axis=1)[None, :]
    return np.argmin(distancias, axis=1)


def _clusteres_manzana(imagen, etiquetas, k, seleccion, margen):
    """Devuelve los índices de los clústeres que se consideran manzana."""
    cr = extraer_canal(imagen, "Cr").reshape(-1).astype(np.float64)
    b = extraer_canal(imagen, "b").reshape(-1).astype(np.float64)
    cuenta = np.bincount(etiquetas, minlength=k)
    validos = cuenta > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        cr_medio = np.bincount(etiquetas, weights=cr, minlength=k) / cuenta
        b_medio = np.bincount(etiquetas, weights=b, minlength=k) / cuenta
    cr_medio[~validos] = -np.inf
    b_medio[~validos] = -np.inf

    if seleccion == "max_Cr":
        return [int(np.argmax(cr_medio))]
    if seleccion == "max_suma":
        return [int(np.argmax(cr_medio + b_medio))]
    if seleccion == "max_Cr_o_max_b":
        return sorted({int(np.argmax(cr_medio)), int(np.argmax(b_medio))})
    if seleccion == "relativa":
        # Clústeres cuyo color medio destaca sobre el de toda la imagen.
        destaca = (cr_medio - cr.mean() > margen) | (b_medio - b.mean() > margen)
        return [int(i) for i in np.flatnonzero(destaca & validos)]
    raise ValueError(f"selección desconocida '{seleccion}'. Opciones: {SELECCIONES}")


def kmeans(imagen, caracteristicas="Cr_b", k=4, seleccion="max_Cr", margen=20,
           muestras=20000, intentos=3, iteraciones=20, semilla=0):
    """Segmentación con K-means sobre el color de los píxeles.

    caracteristicas: "bgr", "lab", "ab" o "Cr_b" (canales de cada vector)
    k:               número de clústeres
    seleccion:       qué clústeres son manzana, según su color medio:
                       "max_Cr"          el más rojo (mayor Cr medio)
                       "max_suma"        el de mayor Cr + b* medio
                       "max_Cr_o_max_b"  el más rojo y el más amarillo
                       "relativa"        todos los que superan la media de la
                                         imagen en Cr o en b* por más de `margen`
    margen:          solo para "relativa"
    muestras:        píxeles usados para calcular los centroides
    intentos:        repeticiones con distinta inicialización aleatoria
    iteraciones:     máximo de iteraciones de cada intento
    semilla:         para que el resultado sea reproducible
    """
    if k < 2:
        raise ValueError(f"k debe ser >= 2, no {k}")
    alto, ancho = imagen.shape[:2]
    vectores = _vectores(imagen, caracteristicas)

    # 1-2. Centroides con una muestra aleatoria de píxeles.
    rng = np.random.default_rng(semilla)
    n = vectores.shape[0]
    indices = rng.choice(n, size=min(muestras, n), replace=False)
    cv.setRNGSeed(semilla)
    criterio = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, iteraciones, 0.5)
    _, _, centros = cv.kmeans(vectores[indices], k, None, criterio, intentos, cv.KMEANS_RANDOM_CENTERS)

    # Asignación de TODOS los píxeles a su centroide más cercano.
    etiquetas = _asignar(vectores, centros)

    # 3. Clústeres que son manzana.
    elegidos = _clusteres_manzana(imagen, etiquetas, k, seleccion, margen)
    mascara = np.isin(etiquetas, elegidos)
    return np.where(mascara, 255, 0).astype(np.uint8).reshape(alto, ancho)