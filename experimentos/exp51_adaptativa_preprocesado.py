"""Umbralización adaptativa, fase 2: efecto del preprocesado.

Las dos mejores configuraciones de exp50 (elegidas sobre TRAIN) se prueban con
los cuatro preprocesados comunes P0-P3. Se lleva la mejor de cada estrategia,
en vez de las dos primeras de la tabla, porque esas dos solo se diferencian en
el tamaño de bloque (301 frente a 201, Dice 0.535 frente a 0.528):
    - OR:  media, Cr OR b*, bloque 301, C = -30          (Dice train 0.535)
    - AND: Niblack, Cr AND b*, bloque 401, k = 0.5, C = -5 (Dice train 0.493)

Ejecutar:
    uv run python scripts/ejecutar.py exp51 --conjunto train --sin-figuras
    uv run python scripts/resumen.py --conjunto train --filtro exp51
La mejor combinación se copia a exp52_adaptativa_postprocesado.py.
"""

from src.preprocesado.variantes import PREPROCESADOS

MEJORES = {
    "media_Cr_or_b_b301_C-30": ("adaptativa_combinada", {
        "canales": ("Cr", "b"), "operacion": "or", "metodo": "media", "bloque": 301, "C": -30}),
    "niblack_Cr_and_b_b401_k0.5": ("adaptativa_combinada", {
        "canales": ("Cr", "b"), "operacion": "and", "metodo": "niblack", "bloque": 401, "k": 0.5, "C": -5}),
}

EXPERIMENTOS = [
    {
        "nombre": f"exp51_adapt_{id_modelo}_{id_pre}",
        "descripcion": f"Adaptativa {id_modelo} con preprocesado {id_pre}",
        "preprocesado": pasos,
        "modelo": modelo,
    }
    for id_modelo, modelo in MEJORES.items()
    for id_pre, pasos in PREPROCESADOS.items()
]