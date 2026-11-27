"""Busca hiperparámetros con Optuna (validación cruzada, 5 folds sobre train) y evalúa la mejor sobre TEST.

Cada archivo busquedas_optuna/busXX_*.py define una función espacio(trial) que
devuelve un experimento (ver src/optimizacion.py). Mismo reparto de números
que en experimentos/: Persona 1 bus10-29, Persona 2 bus30-49, Persona 3 bus50-69.

Uso, desde la raíz del proyecto:
    uv run python scripts/optimizar.py --listar                 # búsquedas definidas
    uv run python scripts/optimizar.py bus50 --pruebas 150      # buscar sobre train
    uv run python scripts/optimizar.py bus50 --pruebas 50       # reanuda: 50 pruebas MÁS
    uv run python scripts/optimizar.py bus50 --nuevo            # borra el estudio y empieza de cero
    uv run python scripts/optimizar.py bus50 --solo-guardar     # rehacer CSV y figuras sin buscar
    uv run python scripts/optimizar.py bus50 --evaluar-test     # mejor config -> test (al final)

Una búsqueda puede definir PUNTOS_DE_PARTIDA: combinaciones ya conocidas que
Optuna prueba primero, para no acabar peor que ellas. Cada una puede ser un
diccionario escrito a mano (con los mismos nombres de parámetros que espacio)
o el nombre de otra búsqueda ("bus50_adaptativa"), y entonces se lee su mejor
combinación de busquedas_optuna/mejores/ al lanzar.

Salidas:
    resultados/optuna/estudios.db              todas las búsquedas (para reanudarlas)
    resultados/optuna/<busqueda>/pruebas.csv   una fila por prueba, ordenada por Dice
    resultados/optuna/<busqueda>/*.png         historia, Dice frente a cada parámetro, importancia
    busquedas_optuna/mejores/<busqueda>.json          la mejor configuración (SÍ se sube a Git)
    resultados/<busqueda>_mejor/               evaluación sobre test, como cualquier experimento
"""

import argparse
import importlib.util
import sys
import time
import warnings
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import optuna  # noqa: E402

from src import evaluacion, optimizacion  # noqa: E402

DIR_BUSQUEDAS = RAIZ / "busquedas_optuna"


def buscar_archivo(prefijo):
    archivos = sorted(DIR_BUSQUEDAS.glob("bus*.py"))
    coincidencias = [p for p in archivos if p.stem == prefijo or p.stem.startswith(prefijo + "_")]
    if len(coincidencias) != 1:
        opciones = ", ".join(p.stem for p in archivos) or "ninguna"
        raise FileNotFoundError(f"'{prefijo}' debe coincidir con una sola búsqueda. Hay: {opciones}")
    return coincidencias[0]


def cargar_busqueda(ruta):
    """Módulo de la búsqueda: espacio(trial) y, opcionalmente, PUNTOS_DE_PARTIDA."""
    spec = importlib.util.spec_from_file_location(ruta.stem, ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    if not hasattr(modulo, "espacio"):
        raise ValueError(f"{ruta.name} no define la función espacio(trial)")
    return modulo


def preparar_puntos(modulo):
    """Puntos de partida de la búsqueda, comprobados contra su espacio.

    Cada elemento de PUNTOS_DE_PARTIDA puede ser:
      - un diccionario de parámetros escrito a mano (p. ej. lo mejor de la rejilla), o
      - el nombre de otra búsqueda (p. ej. "bus50_adaptativa"): se lee su mejor
        combinación de busquedas_optuna/mejores/<nombre>.json en el momento de lanzar,
        así que siempre está actualizada.
    Un punto que no encaja en el espacio (falta un parámetro o un valor no está
    permitido) se omite con un aviso, en vez de hacer fallar la búsqueda.
    """
    puntos = []
    for punto in getattr(modulo, "PUNTOS_DE_PARTIDA", []):
        if isinstance(punto, str):
            origen = f"mejor de {punto}"
            try:
                salida, _ = optimizacion.cargar_mejor(punto)
            except FileNotFoundError:
                print(f"  aviso: no existe busquedas_optuna/mejores/{punto}.json; se omite ese punto de partida")
                continue
            parametros = salida["parametros"]
        else:
            origen = "escrito a mano"
            parametros = punto
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")  # un valor fuera de rango también cuenta como error
                modulo.espacio(optuna.trial.FixedTrial(parametros))
        except Exception as error:  # noqa: BLE001
            print(f"  aviso: el punto de partida ({origen}) no encaja en esta búsqueda: {error}. Se omite.")
            continue
        puntos.append((origen, parametros))
    return puntos


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("busqueda", nargs="?", help="prefijo como bus50")
    parser.add_argument("--pruebas", type=int, default=100, help="número de pruebas NUEVAS (por defecto 100)")
    parser.add_argument("--semilla", type=int, default=0)
    parser.add_argument("--sin-poda", action="store_true", help="evaluar todas las pruebas enteras")
    parser.add_argument("--nuevo", action="store_true", help="borrar el estudio guardado y empezar de cero")
    parser.add_argument("--solo-guardar", action="store_true", help="no buscar; rehacer CSV, JSON y figuras")
    parser.add_argument("--evaluar-test", action="store_true", help="evaluar la mejor configuración sobre test")
    parser.add_argument("--listar", action="store_true")
    args = parser.parse_args()

    if args.listar:
        for ruta in sorted(DIR_BUSQUEDAS.glob("bus*.py")):
            print(ruta.stem)
        return
    if not args.busqueda:
        parser.print_help()
        return

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    ruta = buscar_archivo(args.busqueda)
    nombre = ruta.stem

    if args.evaluar_test:
        salida, config = optimizacion.cargar_mejor(nombre)
        config["nombre"] = f"{nombre}_mejor"
        dice_cv = salida.get("dice_cv", salida.get("dice_train"))
        config["descripcion"] = (f"Mejor configuración de Optuna ({salida['pruebas_completas']} pruebas, "
                                 f"Dice validación cruzada {dice_cv:.3f})")
        evaluacion.ejecutar(config, conjunto="test", guardar_figuras=True)
        print(f"\nTabla: uv run python scripts/resumen.py --filtro {nombre}")
        return

    estudio = optimizacion.crear_estudio(nombre, semilla=args.semilla, podar=not args.sin_poda, nuevo=args.nuevo)

    if not args.solo_guardar:
        modulo = cargar_busqueda(ruta)
        espacio = modulo.espacio
        previas = len(estudio.trials)
        # Puntos de partida: combinaciones ya conocidas (p. ej. lo mejor de la rejilla)
        # que Optuna prueba primero. Solo al crear el estudio, para no repetirlas al reanudar.
        if previas == 0 and getattr(modulo, "PUNTOS_DE_PARTIDA", []):
            puntos = preparar_puntos(modulo)
            for origen, parametros in puntos:
                estudio.enqueue_trial(parametros, skip_if_exists=True)
                print(f"  punto de partida ({origen}): {parametros}")
            print(f"{len(puntos)} punto(s) de partida en cola: se prueban primero\n")
        datos_mem = optimizacion.DatosEnMemoria()
        print(f"Búsqueda {nombre}: {previas} pruebas previas, {args.pruebas} nuevas "
              f"(validación cruzada sobre train; el test no se usa)\n")

        def informar(estudio, prueba):
            if prueba.state == optuna.trial.TrialState.PRUNED:
                hechos = max(prueba.intermediate_values)
                estado = f"podada tras {hechos} fold{'s' if hechos > 1 else ''}"
            else:
                estado = f"Dice {prueba.value:.3f} ±{prueba.user_attrs.get('std_dice_folds', 0):.3f}"
            try:
                mejor = f"{estudio.best_value:.3f}"
            except ValueError:
                mejor = "-"
            print(f"  prueba {prueba.number:4d}  {estado:18s}  mejor {mejor}  {prueba.params}", flush=True)

        inicio = time.perf_counter()
        estudio.optimize(optimizacion.objetivo(espacio, datos_mem), n_trials=args.pruebas,
                         callbacks=[informar], gc_after_trial=False)
        print(f"\n{args.pruebas} pruebas en {time.perf_counter() - inicio:.0f} s")

    salida = optimizacion.guardar_resultados(estudio)
    m = salida["metricas_cv"]
    print(f"\nMejor prueba: {salida['mejor_prueba']}  Dice (media de los folds) {salida['dice_cv']:.3f} "
          f"± {m['std_dice_folds']:.3f} entre folds")
    print(f"  Dice por fold: {', '.join(f'{d:.3f}' for d in salida['dice_por_fold'])}")
    print(f"  IoU {m['media_iou']:.3f}  precisión {m['media_precision']:.3f}  recall {m['media_recall']:.3f}")
    print(f"  preprocesado:  {salida['config']['preprocesado']}")
    print(f"  modelo:        {salida['config']['modelo']}")
    print(f"  postprocesado: {salida['config']['postprocesado']}")
    print(f"\n{salida['pruebas_completas']} completas, {salida['pruebas_podadas']} podadas.")
    print(f"Figuras y CSV en resultados/optuna/{nombre}/ ; configuración en busquedas_optuna/mejores/{nombre}.json")
    print(f"Cuando no vayas a cambiar nada más: uv run python scripts/optimizar.py {nombre} --evaluar-test")


if __name__ == "__main__":
    main()
