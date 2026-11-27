"""Análisis del preprocesado: figuras y tablas para la memoria.

Uso, desde la raíz del proyecto:
    uv run python scripts/analizar_preprocesado.py
    uv run python scripts/analizar_preprocesado.py --imagen img_094

Genera en resultados/preprocesado/:
    1_variantes_<imagen>.png   La misma imagen con cada paso de preprocesado.
    2_histogramas_canales.png  Histograma de manzana frente a fondo por canal,
                               sumando todo el dataset.
    3_separabilidad.csv        Separabilidad manzana/fondo de cada canal con
                               cada preprocesado (razón de Fisher).
    3_separabilidad.png        La misma tabla como mapa de calor.
    4_perfil_vertical.png      Fracción de píxeles de manzana por altura de la
                               imagen, para decidir si tiene sentido una ROI.

Razón de Fisher: (media_manzana - media_fondo)^2 / (var_manzana + var_fondo).
Cuanto mayor, más fácil es separar manzana y fondo con un umbral sobre ese
canal. Sirve para comparar preprocesados y canales con un único número.
Limitación: en el canal H el rojo está en los dos extremos (0 y 179), así que
para las manzanas rojas la media de H es engañosa.
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2 as cv
import matplotlib

matplotlib.use("Agg")  # guarda las figuras sin abrir ventanas
import matplotlib.pyplot as plt
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.preprocesado import aplicar, canal, describir  # noqa: E402

# Solo imágenes de train: el test del profesor no se mira para decidir nada.
DIR_IMAGENES = RAIZ / "data" / "train" / "imagenes"
DIR_MASCARAS = RAIZ / "data" / "train" / "mascaras"
DIR_SALIDA = RAIZ / "resultados" / "preprocesado"

CANALES = ["gris", "H", "S", "V", "a", "b", "Cr"]

# Preprocesados que se comparan. Para probar otro, basta con añadir una línea.
VARIANTES = {
    "original": [],
    "gaussiano 5": [("gaussiano", {"k": 5})],
    "mediana 5": [("mediana", {"k": 5})],
    "bilateral": [("bilateral", {"d": 9, "sigma_color": 75, "sigma_espacio": 75})],
    "ecualizar": [("ecualizar", {})],
    "clahe 2.0": [("clahe", {"clip": 2.0, "rejilla": 8})],
    "gamma 0.7": [("gamma", {"valor": 0.7})],
    "estirar 1-99": [("estirar", {"p_bajo": 1, "p_alto": 99})],
    "gaussiano + clahe": [("gaussiano", {"k": 5}), ("clahe", {"clip": 2.0, "rejilla": 8})],
    "bilateral + clahe": [
        ("bilateral", {"d": 9, "sigma_color": 75, "sigma_espacio": 75}),
        ("clahe", {"clip": 2.0, "rejilla": 8}),
    ],
}


def cargar_parejas():
    """Lista de (nombre, imagen BGR, máscara booleana) de todo el dataset."""
    parejas = []
    for ruta in sorted(DIR_IMAGENES.glob("*.png")):
        imagen = cv.imread(str(ruta), cv.IMREAD_COLOR)
        mascara = cv.imread(str(DIR_MASCARAS / ruta.name), cv.IMREAD_GRAYSCALE)
        if imagen is None or mascara is None:
            raise FileNotFoundError(f"no se pudo leer la pareja {ruta.name}")
        parejas.append((ruta.stem, imagen, mascara > 127))
    if not parejas:
        raise FileNotFoundError(f"no hay imágenes en {DIR_IMAGENES}")
    return parejas


def figura_variantes(nombre, imagen):
    """Figura 1: la misma imagen con cada preprocesado."""
    n = len(VARIANTES)
    columnas = 5
    filas = int(np.ceil(n / columnas))
    fig, ejes = plt.subplots(filas, columnas, figsize=(3.2 * columnas, 5.6 * filas))
    for eje, (titulo, pasos) in zip(ejes.flat, VARIANTES.items()):
        eje.imshow(cv.cvtColor(aplicar(imagen, pasos), cv.COLOR_BGR2RGB))
        eje.set_title(titulo, fontsize=11)
        eje.axis("off")
    for eje in list(ejes.flat)[n:]:
        eje.axis("off")
    fig.suptitle(f"Variantes de preprocesado ({nombre})", fontsize=14)
    fig.tight_layout()
    fig.savefig(DIR_SALIDA / f"1_variantes_{nombre}.png", dpi=110)
    plt.close(fig)


def figura_histogramas(parejas):
    """Figura 2: histograma normalizado de manzana y de fondo para cada canal."""
    hist_manzana = {c: np.zeros(256) for c in CANALES}
    hist_fondo = {c: np.zeros(256) for c in CANALES}
    for _, imagen, mascara in parejas:
        for c in CANALES:
            valores = canal(imagen, c)
            hist_manzana[c] += np.bincount(valores[mascara], minlength=256)
            hist_fondo[c] += np.bincount(valores[~mascara], minlength=256)

    fig, ejes = plt.subplots(2, 4, figsize=(18, 7))
    for eje, c in zip(ejes.flat, CANALES):
        eje.plot(hist_fondo[c] / hist_fondo[c].sum(), color="tab:green", label="fondo")
        eje.plot(hist_manzana[c] / hist_manzana[c].sum(), color="tab:red", label="manzana")
        eje.set_title(f"Canal {c}")
        eje.set_xlim(0, 180 if c == "H" else 255)
        eje.set_xlabel("valor")
        eje.legend()
        eje.grid(alpha=0.3)
    ejes.flat[-1].axis("off")
    fig.suptitle(f"Histogramas normalizados manzana frente a fondo ({len(parejas)} imágenes)")
    fig.tight_layout()
    fig.savefig(DIR_SALIDA / "2_histogramas_canales.png", dpi=110)
    plt.close(fig)


def fisher(valores, mascara):
    """Razón de Fisher entre los píxeles de manzana y los de fondo."""
    manzana = valores[mascara].astype(np.float64)
    fondo = valores[~mascara].astype(np.float64)
    if manzana.size == 0:
        return np.nan
    return (manzana.mean() - fondo.mean()) ** 2 / (manzana.var() + fondo.var() + 1e-9)


def tabla_separabilidad(parejas):
    """Tabla 3: razón de Fisher media (sobre las imágenes) por preprocesado y canal."""
    tabla = np.zeros((len(VARIANTES), len(CANALES)))
    for i, (titulo, pasos) in enumerate(VARIANTES.items()):
        print(f"  separabilidad: {titulo}")
        acumulado = np.zeros(len(CANALES))
        for _, imagen, mascara in parejas:
            pre = aplicar(imagen, pasos)
            acumulado += [fisher(canal(pre, c), mascara) for c in CANALES]
        tabla[i] = acumulado / len(parejas)

    with open(DIR_SALIDA / "3_separabilidad.csv", "w", newline="") as f:
        escritor = csv.writer(f)
        escritor.writerow(["variante", "pasos"] + CANALES)
        for (titulo, pasos), fila in zip(VARIANTES.items(), tabla):
            escritor.writerow([titulo, describir(pasos)] + [f"{v:.4f}" for v in fila])

    fig, eje = plt.subplots(figsize=(10, 0.55 * len(VARIANTES) + 1.5))
    imagen = eje.imshow(tabla, cmap="viridis", aspect="auto")
    eje.set_xticks(range(len(CANALES)), CANALES)
    eje.set_yticks(range(len(VARIANTES)), list(VARIANTES))
    for (i, j), valor in np.ndenumerate(tabla):
        color = "black" if valor > tabla.max() * 0.6 else "white"
        eje.text(j, i, f"{valor:.2f}", ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(imagen, label="razón de Fisher (mayor = más separable)")
    eje.set_title("Separabilidad manzana/fondo por canal y preprocesado (media por imagen)")
    fig.tight_layout()
    fig.savefig(DIR_SALIDA / "3_separabilidad.png", dpi=110)
    plt.close(fig)
    return tabla


def figura_perfil_vertical(parejas):
    """Figura 4: fracción de píxeles de manzana en cada fila de la imagen."""
    alto = parejas[0][2].shape[0]
    suma = np.zeros(alto)
    maxima_fila = 0.0
    for _, _, mascara in parejas:
        filas_con_manzana = np.nonzero(mascara.any(axis=1))[0]
        if filas_con_manzana.size:
            maxima_fila = max(maxima_fila, filas_con_manzana.max() / alto)
        suma += mascara.mean(axis=1)
    perfil = suma / len(parejas)

    acumulado = np.cumsum(perfil) / perfil.sum()
    y_99 = np.searchsorted(acumulado, 0.99) / alto
    y_100 = np.searchsorted(acumulado, 1.0 - 1e-12) / alto

    fig, eje = plt.subplots(figsize=(10, 4))
    eje.plot(np.arange(alto) / alto, perfil, color="tab:red")
    eje.axvline(y_99, color="gray", linestyle="--", label=f"99 % de los píxeles de manzana: y < {y_99:.2f}")
    eje.axvline(maxima_fila, color="black", linestyle=":", label=f"fila más baja con manzana: y = {maxima_fila:.2f}")
    eje.set_xlabel("altura relativa en la imagen (0 = arriba, 1 = abajo)")
    eje.set_ylabel("fracción media de manzana")
    eje.set_title(f"Dónde están las manzanas en vertical ({len(parejas)} imágenes)")
    eje.legend()
    eje.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(DIR_SALIDA / "4_perfil_vertical.png", dpi=110)
    plt.close(fig)
    return y_99, y_100, maxima_fila


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--imagen", default=None, help="nombre de la imagen para la figura 1 (por ejemplo img_094)")
    args = parser.parse_args()

    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    print("Cargando el dataset...")
    parejas = cargar_parejas()
    print(f"  {len(parejas)} parejas imagen/máscara")

    nombre, imagen, _ = next(
        (p for p in parejas if p[0] == args.imagen), parejas[0]
    ) if args.imagen else parejas[0]
    print(f"Figura 1: variantes de preprocesado sobre {nombre}")
    figura_variantes(nombre, imagen)

    print("Figura 2: histogramas por canal")
    figura_histogramas(parejas)

    print("Tabla 3: separabilidad (es la parte lenta)")
    tabla = tabla_separabilidad(parejas)

    print("Figura 4: perfil vertical")
    y_99, _, maxima_fila = figura_perfil_vertical(parejas)

    print("\nResumen")
    mejor = np.unravel_index(np.nanargmax(tabla), tabla.shape)
    print(f"  Mejor combinación: {list(VARIANTES)[mejor[0]]} + canal {CANALES[mejor[1]]} "
          f"(Fisher = {tabla[mejor]:.2f})")
    original = tabla[0]
    for j, c in enumerate(CANALES):
        mejor_i = int(np.nanargmax(tabla[:, j]))
        print(f"  Canal {c:4s}: original {original[j]:.2f} -> mejor {tabla[mejor_i, j]:.2f} con {list(VARIANTES)[mejor_i]}")
    print(f"  El 99 % de los píxeles de manzana está por encima de y = {y_99:.2f}; "
          f"la fila más baja con manzana está en y = {maxima_fila:.2f}")
    print(f"\nResultados en {DIR_SALIDA.relative_to(RAIZ)}/")


if __name__ == "__main__":
    main()
