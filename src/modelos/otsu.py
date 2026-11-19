"""Umbralización de Otsu (Tema 5) - Persona 2.

Firma común: metodo(imagen_bgr, **params) -> máscara uint8 (0 fondo, 255 manzana).

Algoritmo de Otsu (implementado aquí con NumPy; da el mismo umbral que
cv.threshold con THRESH_OTSU):
    1. Histograma normalizado del canal: p(i) para i = 0..255.
    2. Para cada umbral candidato t, dos clases: [0, t] y [t+1, 255], con
       pesos w0, w1 y medias mu0, mu1.
    3. Varianza interclase: sigma_b^2(t) = w0 * w1 * (mu0 - mu1)^2.
    4. El umbral es el t que la maximiza.

Problema en este dataset: Otsu supone dos clases de tamaño parecido, pero el
fondo ocupa ~95 % de la imagen y tiene varios modos (cielo, hojas, hierba).
El umbral clásico acaba separando unos modos del fondo de otros, y marca
como manzana mucho fondo (precisión muy baja). Por eso se prueban las dos
extensiones de la teoría:
    - multi-Otsu: 3 clases con 2 umbrales; la manzana es la clase superior.
    - Otsu iterado (la "mejora de Otsu"): se vuelve a calcular el umbral solo
      con los píxeles por encima del anterior, en el rango [T + 1, 255].
"""

import numpy as np

from src.preprocesado import canal as extraer_canal

VARIANTES = ("clasico", "multi", "iterado")


def _histograma(canal_2d, minimo=0):
    """Histograma de 256 niveles, ignorando los valores <= minimo - 1."""
    hist = np.bincount(canal_2d.ravel(), minlength=256).astype(np.float64)
    hist[:minimo] = 0
    return hist


def umbral_otsu(hist):
    """Umbral de Otsu de un histograma: maximiza la varianza interclase.

    Devuelve t: la clase superior son los niveles > t (igual que OpenCV).
    """
    total = hist.sum()
    if total == 0:
        return 0
    p = hist / total
    niveles = np.arange(256)
    w0 = np.cumsum(p)                  # peso de la clase [0, t]
    mu_acum = np.cumsum(p * niveles)   # suma de i * p(i) en [0, t]
    mu_total = mu_acum[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        # Forma equivalente de w0 * w1 * (mu0 - mu1)^2, sin dividir dos veces.
        varianza_inter = (mu_total * w0 - mu_acum) ** 2 / (w0 * (1 - w0))
    varianza_inter = np.nan_to_num(varianza_inter, nan=-1.0, posinf=-1.0)
    return int(np.argmax(varianza_inter))


def umbrales_multiotsu(hist):
    """Multi-Otsu con 3 clases: devuelve (t1, t2).

    Las clases son [0, t1), [t1, t2) y [t2, 255]. Se prueban todas las parejas
    t1 < t2 a la vez con sumas acumuladas, y se elige la que maximiza la
    varianza interclase, que con la media total fija equivale a maximizar
    sum_k (suma de i*p en la clase k)^2 / (peso de la clase k).
    """
    total = hist.sum()
    if total == 0:
        return 0, 0
    p = hist / total
    # P[k] y M[k]: peso y suma de i*p de los niveles 0..k-1.
    P = np.concatenate(([0.0], np.cumsum(p)))
    M = np.concatenate(([0.0], np.cumsum(p * np.arange(256))))

    t1 = np.arange(1, 255)[:, None]
    t2 = np.arange(2, 256)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        termino = lambda a, b: np.where(P[b] - P[a] > 0, (M[b] - M[a]) ** 2 / (P[b] - P[a]), 0.0)
        objetivo = termino(0, t1) + termino(t1, t2) + termino(t2, 256)
    objetivo = np.where(t2 > t1, objetivo, -np.inf)
    i, j = np.unravel_index(np.argmax(objetivo), objetivo.shape)
    return int(t1[i, 0]), int(t2[0, j])


def umbral(canal_2d, variante="clasico", pasos=2):
    """Umbral para un canal 2D. Los píxeles de manzana son los >= umbral."""
    if variante == "clasico":
        return umbral_otsu(_histograma(canal_2d)) + 1
    if variante == "multi":
        return umbrales_multiotsu(_histograma(canal_2d))[1]
    if variante == "iterado":
        t = -1
        for _ in range(pasos):
            hist = _histograma(canal_2d, minimo=t + 1)
            if hist.sum() == 0:
                break
            t = umbral_otsu(hist)
        return t + 1
    raise ValueError(f"variante desconocida '{variante}'. Opciones: {VARIANTES}")


def _segmentar_canal(imagen, canal, variante, pasos, invertir):
    valores = extraer_canal(imagen, canal)
    if invertir:
        valores = 255 - valores  # la manzana pasa a ser la parte alta del histograma
    t = umbral(valores, variante, pasos)
    return np.where(valores >= t, 255, 0).astype(np.uint8)


def otsu(imagen, canal="Cr", variante="clasico", pasos=2, invertir=False):
    """Segmenta un único canal con Otsu.

    canal:    "gris", "Cr", "a", "b", "H", "S", "V"... (ver src.preprocesado.color)
    variante: "clasico" (2 clases), "multi" (3 clases, se queda la superior)
              o "iterado" (repite Otsu `pasos` veces sobre la parte alta).
    invertir: True si la manzana es MÁS OSCURA que el fondo en ese canal
              (por ejemplo H, donde las manzanas tienen menor tono que las hojas).
    """
    return _segmentar_canal(imagen, canal, variante, pasos, invertir)


def otsu_combinado(imagen, canales=("Cr", "b"), operacion="and", variante="multi", pasos=2, invertir=()):
    """Otsu en varios canales y combinación de las máscaras.

    Motivo: Cr separa bien las manzanas rojas y b* las amarillas, pero ninguno
    sirve para las dos. "and" se queda con lo que ambos marcan (más precisión);
    "or" con lo que marque cualquiera (más recall).
    invertir: canales a invertir, por ejemplo ("H",).
    """
    if operacion not in ("and", "or"):
        raise ValueError("operacion debe ser 'and' u 'or'")
    mascaras = [_segmentar_canal(imagen, c, variante, pasos, c in invertir) for c in canales]
    resultado = mascaras[0]
    for m in mascaras[1:]:
        resultado = np.bitwise_and(resultado, m) if operacion == "and" else np.bitwise_or(resultado, m)
    return resultado
