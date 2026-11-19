"""Métricas de segmentación (Tema 5; Práctica 7), implementadas con NumPy.

Se comparan dos máscaras del mismo tamaño, la predicha y la de referencia,
píxel a píxel. Cualquier valor distinto de 0 cuenta como manzana.

    TP: manzana predicha que es manzana      FP: manzana predicha que es fondo
    FN: fondo predicho que es manzana        TN: fondo predicho que es fondo

Métricas:
    exactitud      (TP + TN) / total     Engañosa aquí: el fondo es ~95 % de la imagen,
                                         así que predecir "todo fondo" ya da ~0.95.
    precision      TP / (TP + FP)        De lo que marco como manzana, cuánto lo es.
    recall         TP / (TP + FN)        De las manzanas reales, cuánto encuentro.
    especificidad  TN / (TN + FP)        Del fondo real, cuánto dejo como fondo.
    fpr            FP / (FP + TN)        Tasa de falsas alarmas (1 - especificidad).
    iou            TP / (TP + FP + FN)   Índice de Jaccard.
    dice           2TP / (2TP + FP + FN) Equivale al F1 a nivel de píxel.

Casos límite: si un cociente es 0 / 0 (por ejemplo, una imagen sin manzanas y
una predicción vacía), se devuelve 1.0 para iou, dice, precision y recall,
porque la predicción es perfecta. Esto no ocurre en el dataset actual, pero así
ninguna métrica sale como NaN.
"""

import numpy as np

NOMBRES = ("exactitud", "precision", "recall", "especificidad", "fpr", "iou", "dice")


def matriz_confusion(prediccion, referencia):
    """Devuelve (TP, FP, FN, TN) como enteros de Python."""
    if prediccion.shape != referencia.shape:
        raise ValueError(f"tamaños distintos: predicción {prediccion.shape}, referencia {referencia.shape}")
    pred = prediccion > 0
    ref = referencia > 0
    tp = int(np.count_nonzero(pred & ref))
    fp = int(np.count_nonzero(pred & ~ref))
    fn = int(np.count_nonzero(~pred & ref))
    tn = int(pred.size - tp - fp - fn)
    return tp, fp, fn, tn


def _cociente(numerador, denominador, si_cero=1.0):
    return numerador / denominador if denominador > 0 else si_cero


def desde_confusion(tp, fp, fn, tn):
    """Calcula todas las métricas a partir de la matriz de confusión."""
    return {
        "exactitud": _cociente(tp + tn, tp + fp + fn + tn),
        "precision": _cociente(tp, tp + fp),
        "recall": _cociente(tp, tp + fn),
        "especificidad": _cociente(tn, tn + fp),
        "fpr": _cociente(fp, fp + tn, si_cero=0.0),
        "iou": _cociente(tp, tp + fp + fn),
        "dice": _cociente(2 * tp, 2 * tp + fp + fn),
    }


def calcular(prediccion, referencia):
    """Métricas de una imagen, más su matriz de confusión (tp, fp, fn, tn)."""
    tp, fp, fn, tn = matriz_confusion(prediccion, referencia)
    resultado = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    resultado.update(desde_confusion(tp, fp, fn, tn))
    return resultado


def agregar(filas):
    """Resume las métricas de varias imágenes de dos formas.

    - media_<m> y std_<m>: media y desviación típica de la métrica por imagen
      (cada imagen pesa lo mismo). Es la que se usa para comparar métodos.
    - global_<m>: métrica calculada sumando TP, FP, FN y TN de todas las
      imágenes (las imágenes con más manzanas pesan más).
    """
    if not filas:
        raise ValueError("no hay filas que agregar")
    resumen = {"n_imagenes": len(filas)}
    for m in NOMBRES:
        valores = np.array([fila[m] for fila in filas], dtype=float)
        resumen[f"media_{m}"] = float(valores.mean())
        resumen[f"std_{m}"] = float(valores.std())
    totales = [sum(fila[k] for fila in filas) for k in ("tp", "fp", "fn", "tn")]
    for m, valor in desde_confusion(*totales).items():
        resumen[f"global_{m}"] = valor
    return resumen
