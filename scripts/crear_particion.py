"""Crea la partición fija de entrenamiento y test en src/particion.json.

Se ejecuta UNA SOLA VEZ, por una sola persona, y el archivo se sube a GitHub.
Si cada uno la generara por su cuenta, evaluaríamos sobre imágenes distintas
y las comparaciones entre métodos no valdrían.

Uso, desde la raíz del proyecto:
    uv run python scripts/crear_particion.py
    uv run python scripts/crear_particion.py --test 0.3 --semilla 42

- train: para ajustar parámetros (finetuning) y entrenar los clasificadores.
- test: SOLO para la evaluación final. Nunca se ajusta nada mirando el test.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.datos import RUTA_PARTICION, listar_disponibles  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", type=float, default=0.3, help="fracción de imágenes para test (por defecto 0.3)")
    parser.add_argument("--semilla", type=int, default=42, help="semilla del generador aleatorio")
    parser.add_argument("--forzar", action="store_true", help="sobrescribir una partición que ya existe")
    args = parser.parse_args()

    if RUTA_PARTICION.exists() and not args.forzar:
        print(f"Ya existe {RUTA_PARTICION.relative_to(RAIZ)}. No se toca para no cambiar la partición del grupo.")
        print("Si de verdad hay que rehacerla, usa --forzar y avisa a todos.")
        return
    if not 0 < args.test < 1:
        raise ValueError("--test debe estar entre 0 y 1")

    disponibles = listar_disponibles()
    generador = np.random.default_rng(args.semilla)
    barajadas = list(generador.permutation(disponibles))
    n_test = max(1, round(len(barajadas) * args.test))

    particion = {
        "semilla": args.semilla,
        "fraccion_test": args.test,
        "train": sorted(barajadas[n_test:]),
        "test": sorted(barajadas[:n_test]),
    }
    with open(RUTA_PARTICION, "w", encoding="utf-8") as f:
        json.dump(particion, f, indent=2, ensure_ascii=False)

    print(f"Partición guardada en {RUTA_PARTICION.relative_to(RAIZ)}: "
          f"{len(particion['train'])} train y {len(particion['test'])} test")


if __name__ == "__main__":
    main()
