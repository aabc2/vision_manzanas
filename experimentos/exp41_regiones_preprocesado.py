"""Crecimiento de regiones, fase 2: efecto del preprocesado.

Las dos mejores configuraciones de exp40 (elegidas sobre TRAIN) se prueban con
los cuatro preprocesados comunes P0-P3.

>>> Actualiza MEJORES con lo que salga de exp40 en train. <<<
Los valores de abajo son los que mejor funcionaron con las imágenes de muestra.

Ejecutar:
    uv run python scripts/ejecutar.py exp41 --conjunto train --sin-figuras
    uv run python scripts/resumen.py --conjunto train --filtro exp41
"""

from src.preprocesado.variantes import PREPROCESADOS

MEJORES = {
    "H_t3_geo100": ("crecimiento", {"color": "H", "tolerancia": 3, "iteraciones": 25, "geometrico": True, "factor": 1.0}),
    "H_t5_geo100": ("crecimiento", {"color": "H", "tolerancia": 5, "iteraciones": 25, "geometrico": True, "factor": 1.0}),
}

EXPERIMENTOS = [
    {
        "nombre": f"exp41_reg_{id_modelo}_{id_pre}",
        "descripcion": f"Crecimiento {id_modelo} con preprocesado {id_pre}",
        "preprocesado": pasos,
        "modelo": modelo,
    }
    for id_modelo, modelo in MEJORES.items()
    for id_pre, pasos in PREPROCESADOS.items()
]
