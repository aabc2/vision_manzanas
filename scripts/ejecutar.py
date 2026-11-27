"""Lanza experimentos definidos en la carpeta experimentos/.

Cada archivo experimentos/expXX_*.py define EXPERIMENTOS (lista de
diccionarios) o EXPERIMENTO (uno solo). El formato está en src/evaluacion.py.

Uso, desde la raíz del proyecto:
    uv run python scripts/ejecutar.py exp01               # todos los de exp01_*.py
    uv run python scripts/ejecutar.py exp01 exp20         # varios archivos
    uv run python scripts/ejecutar.py exp01 --filtro P2   # solo las variantes que contengan "P2"
    uv run python scripts/ejecutar.py todos               # todos los archivos
    uv run python scripts/ejecutar.py exp01 --sin-figuras # más rápido
    uv run python scripts/ejecutar.py exp01 --conjunto train   # ajustar: validación cruzada (5 folds)
    uv run python scripts/ejecutar.py --listar            # ver qué hay definido

Numeración para no pisarnos: A exp01-exp19, B exp20-exp39, C exp40-exp59.
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src import evaluacion  # noqa: E402

DIR_EXPERIMENTOS = RAIZ / "experimentos"


def cargar_archivo(ruta):
    """Importa un archivo de experimentos y devuelve su lista de configuraciones."""
    spec = importlib.util.spec_from_file_location(ruta.stem, ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    if hasattr(modulo, "EXPERIMENTOS"):
        return list(modulo.EXPERIMENTOS)
    if hasattr(modulo, "EXPERIMENTO"):
        return [modulo.EXPERIMENTO]
    raise ValueError(f"{ruta.name} no define EXPERIMENTOS ni EXPERIMENTO")


def buscar(prefijos):
    archivos = sorted(p for p in DIR_EXPERIMENTOS.glob("exp*.py"))
    if "todos" in prefijos:
        return archivos
    elegidos = []
    for prefijo in prefijos:
        coincidencias = [p for p in archivos if p.stem == prefijo or p.stem.startswith(prefijo + "_")]
        if not coincidencias:
            raise FileNotFoundError(f"no hay ningún experimentos/{prefijo}_*.py")
        elegidos.extend(c for c in coincidencias if c not in elegidos)
    return elegidos


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experimentos", nargs="*", help="prefijos como exp01, o 'todos'")
    parser.add_argument("--filtro", default=None, help="ejecutar solo los experimentos cuyo nombre contenga este texto")
    parser.add_argument("--conjunto", default="test", choices=["test", "train"],
                        help="train = validación cruzada en las 100 imágenes (5 folds); test = el del profesor")
    parser.add_argument("--sin-figuras", action="store_true", help="no guardar una figura por imagen")
    parser.add_argument("--listar", action="store_true", help="mostrar los experimentos definidos y salir")
    args = parser.parse_args()

    archivos = buscar(args.experimentos or ["todos"]) if (args.listar or args.experimentos) else []
    if not archivos:
        parser.print_help()
        return

    configs = []
    for ruta in archivos:
        configs.extend(cargar_archivo(ruta))
    if args.filtro:
        configs = [c for c in configs if args.filtro in c["nombre"]]

    nombres = [c["nombre"] for c in configs]
    repetidos = sorted({n for n in nombres if nombres.count(n) > 1})
    if repetidos:
        raise ValueError(f"nombres de experimento repetidos: {repetidos}")
    for config in configs:
        evaluacion.validar(config)

    if args.listar:
        for config in configs:
            print(config["nombre"])
        return
    if not configs:
        print("Ningún experimento coincide con el filtro.")
        return

    inicio = time.perf_counter()
    for config in configs:
        evaluacion.ejecutar(config, conjunto=args.conjunto, guardar_figuras=not args.sin_figuras)
    print(f"\n{len(configs)} experimento(s) en {time.perf_counter() - inicio:.0f} s. "
          "Tabla comparativa: uv run python scripts/resumen.py")


if __name__ == "__main__":
    main()
