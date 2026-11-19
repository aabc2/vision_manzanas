"""Junta los resultados de todos los experimentos en una tabla comparativa.

Uso, desde la raíz del proyecto:
    uv run python scripts/resumen.py                  # todos los experimentos
    uv run python scripts/resumen.py --filtro exp01   # solo los que contengan "exp01"
    uv run python scripts/resumen.py --orden iou      # ordenar por otra métrica

Genera:
    resultados/resumen.csv                             tabla con una fila por experimento
    resultados/figuras_memoria/comparativa_<métrica>.png
Las métricas son la media por imagen ± desviación típica.
"""

import argparse
import csv
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src import evaluacion, visualizacion  # noqa: E402

METRICAS = ("dice", "iou", "precision", "recall", "exactitud", "especificidad")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--filtro", default=None, help="solo experimentos cuyo nombre contenga este texto")
    parser.add_argument("--orden", default="dice", choices=METRICAS, help="métrica para ordenar (por defecto dice)")
    parser.add_argument("--conjunto", default="test", help="solo resultados de este conjunto (por defecto test)")
    args = parser.parse_args()

    resumenes = [r for r in evaluacion.cargar_resumenes() if r["conjunto"] == args.conjunto]
    if args.filtro:
        resumenes = [r for r in resumenes if args.filtro in r["nombre"]]
    if not resumenes:
        print("No hay resultados. Ejecuta antes algún experimento con scripts/ejecutar.py")
        return
    resumenes.sort(key=lambda r: r["resumen"][f"media_{args.orden}"], reverse=True)

    # Tabla CSV.
    ruta_csv = evaluacion.DIR_RESULTADOS / ("resumen.csv" if not args.filtro else f"resumen_{args.filtro}.csv")
    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        cabecera = ["experimento", "modelo", "preprocesado", "postprocesado", "escala", "n_imagenes"]
        for m in METRICAS:
            cabecera += [f"{m}_media", f"{m}_std"]
        escritor.writerow(cabecera + ["tiempo_s_imagen"])
        for r in resumenes:
            s = r["resumen"]
            fila = [r["nombre"], r["modelo"], r["preprocesado"], r["postprocesado"], r["escala"], s["n_imagenes"]]
            for m in METRICAS:
                fila += [f"{s[f'media_{m}']:.4f}", f"{s[f'std_{m}']:.4f}"]
            escritor.writerow(fila + [f"{s['tiempo_medio_s']:.3f}"])

    # Tabla por pantalla.
    ancho = max(len(r["nombre"]) for r in resumenes)
    print(f"\n{'experimento':<{ancho}}  {'dice':>13}  {'iou':>13}  {'precisión':>9}  {'recall':>7}  {'s/img':>6}")
    for r in resumenes:
        s = r["resumen"]
        print(f"{r['nombre']:<{ancho}}  {s['media_dice']:.3f} ± {s['std_dice']:.3f}  "
              f"{s['media_iou']:.3f} ± {s['std_iou']:.3f}  {s['media_precision']:>9.3f}  "
              f"{s['media_recall']:>7.3f}  {s['tiempo_medio_s']:>6.2f}")

    # Figuras.
    sufijo = f"_{args.filtro}" if args.filtro else ""
    etiquetas = [r["nombre"] for r in resumenes]
    for m in ("dice", "iou"):
        fig = visualizacion.figura_barras(
            etiquetas,
            [r["resumen"][f"media_{m}"] for r in resumenes],
            [r["resumen"][f"std_{m}"] for r in resumenes],
            metrica=m,
        )
        visualizacion.guardar(fig, evaluacion.DIR_RESULTADOS / "figuras_memoria" / f"comparativa_{m}{sufijo}.png")

    print(f"\nTabla en {ruta_csv.relative_to(RAIZ)} y figuras en resultados/figuras_memoria/")


if __name__ == "__main__":
    main()
