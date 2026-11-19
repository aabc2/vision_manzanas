"""Otsu, fase 2: efecto del preprocesado.

Las mejores configuraciones de exp30 (elegidas sobre TRAIN) se prueban con
los cuatro preprocesados comunes P0-P3.

>>> Actualiza MEJORES con lo que salga de exp30 en train. <<<
Los valores de abajo son los que mejor funcionaron con las imágenes de muestra.

Ejecutar:
    uv run python scripts/ejecutar.py exp31 --conjunto train --sin-figuras
    uv run python scripts/resumen.py --conjunto train --filtro exp31
"""

from src.preprocesado.variantes import PREPROCESADOS

MEJORES = {
    "iterado_Cr_and_b": ("otsu_combinado", {"canales": ("Cr", "b"), "operacion": "and", "variante": "iterado"}),
    "multi_Cr_and_b": ("otsu_combinado", {"canales": ("Cr", "b"), "operacion": "and", "variante": "multi"}),
}

EXPERIMENTOS = [
    {
        "nombre": f"exp31_otsu_{id_modelo}_{id_pre}",
        "descripcion": f"Otsu {id_modelo} con preprocesado {id_pre}",
        "preprocesado": pasos,
        "modelo": modelo,
    }
    for id_modelo, modelo in MEJORES.items()
    for id_pre, pasos in PREPROCESADOS.items()
]
