"""Funciones de dibujo comunes, para que todas las figuras tengan el mismo estilo.

Estas funciones crean las figuras pero no deciden dónde guardarlas: eso lo
hacen los scripts, llamando a guardar().

Colores de la superposición (los mismos en toda la memoria):
    verde = acierto (TP)   rojo = falso positivo (FP)   azul = falso negativo (FN)
"""

from pathlib import Path

import cv2 as cv
import matplotlib

matplotlib.use("Agg")  # guarda figuras sin abrir ventanas
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLOR_TP = (0, 200, 0)
COLOR_FP = (230, 0, 0)
COLOR_FN = (0, 90, 255)


def a_rgb(imagen_bgr):
    """OpenCV trabaja en BGR y Matplotlib en RGB."""
    return cv.cvtColor(imagen_bgr, cv.COLOR_BGR2RGB)


def superposicion(imagen_bgr, prediccion, referencia, opacidad=0.55):
    """Imagen RGB con los aciertos y errores de la predicción pintados encima."""
    pred = prediccion > 0
    ref = referencia > 0
    salida = a_rgb(imagen_bgr).astype(np.float32)
    for zona, color in ((pred & ref, COLOR_TP), (pred & ~ref, COLOR_FP), (~pred & ref, COLOR_FN)):
        salida[zona] = (1 - opacidad) * salida[zona] + opacidad * np.array(color, dtype=np.float32)
    return salida.astype(np.uint8)


def figura_comparacion(imagen_bgr, referencia, prediccion, titulo="", metricas=None):
    """Original | referencia | predicción | superposición de aciertos y errores."""
    fig, ejes = plt.subplots(1, 4, figsize=(14, 6.2))
    paneles = [
        (a_rgb(imagen_bgr), "Original", None),
        (referencia, "Referencia", "gray"),
        (prediccion, "Predicción", "gray"),
        (superposicion(imagen_bgr, prediccion, referencia), "Verde TP · Rojo FP · Azul FN", None),
    ]
    for eje, (contenido, nombre, cmap) in zip(ejes, paneles):
        eje.imshow(contenido, cmap=cmap, vmin=0 if cmap else None, vmax=255 if cmap else None)
        eje.set_title(nombre, fontsize=11)
        eje.axis("off")
    if metricas:
        texto = "  ·  ".join(f"{k} {metricas[k]:.3f}" for k in ("iou", "dice", "precision", "recall") if k in metricas)
        titulo = f"{titulo}\n{texto}" if titulo else texto
    if titulo:
        fig.suptitle(titulo, fontsize=12)
    fig.tight_layout()
    return fig


def figura_barras(etiquetas, medias, desviaciones=None, metrica="dice", titulo=None):
    """Barras horizontales de una métrica por experimento, ordenadas de mejor a peor."""
    orden = np.argsort(medias)
    etiquetas = [etiquetas[i] for i in orden]
    medias = np.asarray(medias)[orden]
    errores = None if desviaciones is None else np.asarray(desviaciones)[orden]

    fig, eje = plt.subplots(figsize=(10, 0.45 * len(etiquetas) + 1.5))
    eje.barh(etiquetas, medias, xerr=errores, color="tab:red", alpha=0.8, capsize=3)
    for y, valor in enumerate(medias):
        eje.text(valor + 0.01, y, f"{valor:.3f}", va="center", fontsize=9)
    eje.set_xlim(0, 1.05)
    eje.set_xlabel(f"{metrica} medio por imagen (barra de error = desviación típica)")
    eje.set_title(titulo or f"Comparativa de experimentos: {metrica}")
    eje.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def guardar(fig, ruta, dpi=110):
    """Guarda la figura (creando la carpeta si hace falta) y libera la memoria."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=dpi)
    plt.close(fig)
