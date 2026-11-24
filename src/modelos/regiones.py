"""Crecimiento de regiones (Tema 5) - Persona 2.

Firma común: metodo(imagen_bgr, **params) -> máscara uint8 (0 fondo, 255 manzana).

Algoritmo (crecimiento de regiones a partir de semillas):
    1. Semillas: píxeles que casi seguro son manzana. Por defecto, la salida de
       Otsu iterado sobre Cr AND b*, que en test tiene precisión ~0.83 en las
       imágenes con muchas manzanas: lo que marca casi siempre es manzana.
       Cada componente conectada de las semillas es una región.
    2. En cada iteración, cada región mira sus píxeles vecinos (vecindad 8) y
       se queda con los que cumplen:
         - criterio de color: su color se parece al color MEDIO de la región
           (distancia <= tolerancia);
         - criterio geométrico (opcional): están dentro del círculo mínimo que
           envuelve la semilla, ampliado por `factor`. Las manzanas son
           redondas, así que la región no puede salirse de ese círculo.
    3. Se repite hasta que ninguna región crece o se llega a `iteraciones`.

Motivación (análisis de errores de Otsu): Otsu pierde la parte en sombra de
las manzanas porque Cr y b* bajan con el brillo y un umbral global no sirve
para la luz y la sombra a la vez. El crecimiento parte de lo que Otsu ya
encontró y compara con el color de ESA manzana, no con un umbral global.
El tono H es el criterio de color natural para esto: depende poco del brillo.
"""

import cv2 as cv
import numpy as np

from src.preprocesado import canal as extraer_canal

from .otsu import otsu_combinado

COLORES = ("H", "ab", "Crb")
_VECINDAD = np.ones((3, 3), np.uint8)


def _semillas(imagen, metodo, percentil):
    if metodo == "otsu":
        return otsu_combinado(imagen, canales=("Cr", "b"), operacion="and", variante="iterado")
    if metodo == "percentil":
        # Alternativa sin Otsu: el `percentil` % de píxeles más altos en Cr Y en b*.
        cr = extraer_canal(imagen, "Cr")
        b = extraer_canal(imagen, "b")
        umbral_cr = np.percentile(cr, 100 - percentil)
        umbral_b = np.percentile(b, 100 - percentil)
        return np.where((cr >= umbral_cr) & (b >= umbral_b), 255, 0).astype(np.uint8)
    raise ValueError(f"semillas desconocidas '{metodo}'. Opciones: otsu, percentil")


def _caracteristicas(imagen, color):
    """Valores de color de cada píxel para comparar con la media de la región.

    H se representa como un ángulo (coseno y seno) para que la media y la
    distancia sean circulares: el rojo está en los dos extremos (0 y 179).
    """
    if color == "H":
        h = extraer_canal(imagen, "H").astype(np.float32)
        angulo = h * (2 * np.pi / 180)
        return h, np.stack([np.cos(angulo), np.sin(angulo)], axis=-1)
    canales = {"ab": ("a", "b"), "Crb": ("Cr", "b")}[color]
    valores = np.stack([extraer_canal(imagen, c).astype(np.float32) for c in canales], axis=-1)
    return None, valores


def _zona_geometrica(semillas, factor):
    """Unión de los círculos mínimos que envuelven cada semilla, ampliados por factor."""
    zona = np.zeros_like(semillas)
    contornos, _ = cv.findContours(semillas, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for contorno in contornos:
        (x, y), radio = cv.minEnclosingCircle(contorno)
        cv.circle(zona, (int(round(x)), int(round(y))), int(np.ceil(radio * factor)), 255, -1)
    return zona > 0


def crecimiento(imagen, color="H", tolerancia=3, iteraciones=25, geometrico=True, factor=1.0,
                semillas="otsu", percentil=3.0):
    """Crecimiento de regiones desde semillas.

    color:       criterio de similitud: "H" (tono, distancia circular en unidades
                 de OpenCV 0-179), "ab" (a* y b* de Lab) o "Crb" (Cr y b*).
    tolerancia:  distancia máxima entre el píxel y el color medio de la región.
    iteraciones: máximo de pasos de crecimiento (cada paso avanza 1 píxel).
    geometrico:  si True, la región no sale del círculo envolvente de su semilla.
    factor:      ampliación de ese círculo (1.0 = el círculo mínimo).
    semillas:    "otsu" (Otsu iterado Cr AND b*) o "percentil".
    percentil:   % de píxeles más altos en Cr y b* (solo si semillas="percentil").
    """
    if color not in COLORES:
        raise ValueError(f"color desconocido '{color}'. Opciones: {COLORES}")

    inicial = _semillas(imagen, semillas, percentil)
    n, etiquetas = cv.connectedComponents(inicial, connectivity=8)
    if n <= 1 or iteraciones <= 0:
        return inicial
    etiquetas = etiquetas.astype(np.uint16)  # cv.dilate admite uint16

    tono, valores = _caracteristicas(imagen, color)
    planos = [valores[..., i].ravel() for i in range(valores.shape[-1])]
    permitido = _zona_geometrica(inicial, factor) if geometrico else np.ones(inicial.shape, bool)

    for _ in range(iteraciones):
        # Color medio de cada región (la etiqueta 0 es el fondo y no se usa).
        plano_etiquetas = etiquetas.ravel()
        cuenta = np.maximum(np.bincount(plano_etiquetas, minlength=n), 1)
        medias = np.stack([np.bincount(plano_etiquetas, p, n) / cuenta for p in planos], axis=-1)

        # Candidatos: píxeles sin región con algún vecino en una región.
        # La dilatación asigna a cada píxel la etiqueta (máxima) de sus vecinos.
        vecina = cv.dilate(etiquetas, _VECINDAD)
        candidatos = (etiquetas == 0) & (vecina > 0) & permitido
        if not candidatos.any():
            break
        region = vecina[candidatos]

        if color == "H":
            tono_medio = np.mod(np.arctan2(medias[region, 1], medias[region, 0]), 2 * np.pi) * (180 / (2 * np.pi))
            diferencia = np.abs(tono[candidatos] - tono_medio)
            distancia = np.minimum(diferencia, 180 - diferencia)
        else:
            distancia = np.linalg.norm(valores[candidatos] - medias[region], axis=1)

        aceptados = distancia <= tolerancia
        if not aceptados.any():
            break
        posiciones = np.flatnonzero(candidatos.ravel())[aceptados]
        etiquetas.ravel()[posiciones] = region[aceptados]

    return np.where(etiquetas > 0, 255, 0).astype(np.uint8)
