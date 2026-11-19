# Proyecto VC1: detección de manzanas

Segmentación de las manzanas del árbol situado frente al robot con técnicas
clásicas de visión por computador (OpenCV, NumPy, Matplotlib).

## Instalación

```bash
uv sync
```

Descomprimir el dataset (está en el Drive del grupo, no en GitHub) para que quede así:

```
data/imagenes/img_000.png ... img_099.png
data/mascaras/img_000.png ... img_099.png
```

Todo se ejecuta desde la raíz del proyecto con `uv run python ...`.

## Uso rápido

```bash
uv run python scripts/ejecutar.py --listar        # experimentos definidos
uv run python scripts/ejecutar.py exp30           # ejecutar los de experimentos/exp30_*.py
uv run python scripts/resumen.py                  # tabla comparativa de todo lo ejecutado
```

## Estructura

| Carpeta / archivo | Qué contiene |
|---|---|
| `src/datos.py` | Cargar imagen y máscara; partición fija train/test (`src/particion.json`) |
| `src/preprocesado/` | Filtros, contraste, ROI y canales de color. `variantes.py` = preprocesados comunes P0-P3 |
| `src/modelos/` | Los métodos de segmentación, uno por familia, registrados en `__init__.py` |
| `src/postprocesado.py` | Morfología y filtrado de componentes sobre la máscara predicha |
| `src/metricas.py` | Exactitud, precisión, recall, especificidad, FPR, IoU, Dice |
| `src/evaluacion.py` | Ejecuta un experimento: preprocesar → modelo → postprocesar → métricas |
| `src/visualizacion.py` | Funciones de figuras comunes |
| `experimentos/` | Un archivo por experimento, solo con su configuración |
| `scripts/` | Lo que se lanza desde la terminal |
| `resultados/` | Lo que generan los scripts |

## Cómo añadir un método

1. Escribir la función en su archivo de `src/modelos/`, con la firma
   `metodo(imagen_bgr, **params) -> máscara uint8 (0 / 255)` del mismo alto y ancho.
   Los métodos con entrenamiento siguen el formato descrito en `src/modelos/__init__.py`.
2. Registrarla con una línea en `src/modelos/__init__.py`.
3. Crear `experimentos/expXX_nombre.py` (ver los de Otsu, `exp30`-`exp32`, como ejemplo):

```python
from src.preprocesado.variantes import PREPROCESADOS

EXPERIMENTOS = [
    {
        "nombre": f"exp60_kmeans_{id_pre}",
        "preprocesado": pasos,
        "modelo": ("kmeans", {"k": 3, "espacio": "lab"}),
        "postprocesado": [("apertura", {"k": 5})],
        "escala": 0.5,   # opcional: reduce la imagen para métodos lentos
    }
    for id_pre, pasos in PREPROCESADOS.items()
]
```

4. `uv run python scripts/ejecutar.py exp60`

Numeración de experimentos (10 por algoritmo):

| Persona | Algoritmo 1 | Algoritmo 2 |
|---|---|---|
| 1 | Umbralización global: exp10-19 | Watershed: exp20-29 |
| 2 | Otsu: exp30-39 | Crecimiento de regiones: exp40-49 |
| 3 | Umbralización adaptativa: exp50-59 | K-means: exp60-69 |

## Reglas del grupo

- **Ajustar parámetros mirando solo `train`** (`--conjunto train`); `test` es solo para el resultado final.
- **No cambiar `src/particion.json`** ni la firma de las funciones comunes sin avisar.
- **Preprocesados siempre desde `variantes.py`**, nunca escritos a mano.
- Cada uno edita solo sus archivos de `src/modelos/` y `experimentos/`.
- Flujo de ajuste: fase 1 (canal/parámetros) → fase 2 (preprocesado P0-P3) → fase 3 (postprocesado), todo con `--conjunto train`; al final, ejecutar sobre test.

## Resultados de cada experimento

`resultados/<experimento>/`:
- `metricas.csv`: una fila por imagen.
- `resumen.json`: la configuración y las métricas medias.
- `mascaras/`: las predicciones.
- `figuras/`: original | referencia | predicción | aciertos y errores.

`scripts/resumen.py` genera `resultados/resumen.csv` y las figuras comparativas en `resultados/figuras_memoria/`.
