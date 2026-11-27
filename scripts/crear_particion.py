"""Crea los K folds fijos del conjunto de train en src/particion.json.

Se ejecuta UNA SOLA VEZ, por una sola persona, y el archivo se sube a GitHub.
Si cada uno la generara por su cuenta, usaríamos folds distintos y las
comparaciones entre métodos no valdrían.

Uso, desde la raíz del proyecto:
    uv run python scripts/crear_particion.py
    uv run python scripts/crear_particion.py --k 5 --semilla 42

- train (data/train): las imágenes iniciales, repartidas al azar en K folds del
  mismo tamaño. Sirven para elegir hiperparámetros con validación cruzada.
- test (data/test): el conjunto del profesor. No se reparte ni se toca: solo
  sirve para el resultado final.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.datos import RUTA_PARTICION, listar_disponibles  # noqa: E402


def crear_folds(nombres, k, semilla):
    """Baraja los nombres con la semilla y los reparte en k folds (tamaños que difieren como mucho en 1)."""
    barajados = list(np.random.default_rng(semilla).permutation(sorted(nombres)))
    return [sorted(str(n) for n in barajados[i::k]) for i in range(k)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=5, help="número de folds (por defecto 5)")
    parser.add_argument("--semilla", type=int, default=42, help="semilla del generador aleatorio")
    parser.add_argument("--forzar", action="store_true", help="sobrescribir una partición que ya existe")
    args = parser.parse_args()

    if RUTA_PARTICION.exists() and not args.forzar:
        with open(RUTA_PARTICION, encoding="utf-8") as f:
            actual = json.load(f)
        if "folds" in actual:
            print(f"Ya existe {RUTA_PARTICION.relative_to(RAIZ)} con {len(actual['folds'])} folds. "
                  "No se toca para no cambiar la partición del grupo.")
            print("Si de verdad hay que rehacerla, usa --forzar y avisa a todos.")
            return
    if args.k < 2:
        raise ValueError("--k debe ser al menos 2")

    nombres = listar_disponibles("train")
    folds = crear_folds(nombres, args.k, args.semilla)
    particion = {
        "k": args.k,
        "semilla": args.semilla,
        "descripcion": "Folds de validación cruzada sobre data/train. El test (data/test) no se reparte.",
        "folds": folds,
    }
    with open(RUTA_PARTICION, "w", encoding="utf-8") as f:
        json.dump(particion, f, indent=2, ensure_ascii=False)

    print(f"Partición guardada en {RUTA_PARTICION.relative_to(RAIZ)}: {len(nombres)} imágenes en "
          f"{args.k} folds de {', '.join(str(len(f)) for f in folds)}")


if __name__ == "__main__":
    main()
