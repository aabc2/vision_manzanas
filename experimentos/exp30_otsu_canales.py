"""Otsu, fase 1: qué canal y qué variante de Otsu.

Sin preprocesado ni postprocesado, para comparar solo el canal y la variante:
    - variantes: clásico (2 clases), multi-Otsu (3 clases) e iterado (2 pasos).
    - canales: gris (línea base), Cr, a*, b*, S y H (invertido, porque la
      manzana tiene menos tono que las hojas).
    - combinaciones Cr y b* (AND / OR), para cubrir manzanas rojas y amarillas.

Se ejecuta sobre TRAIN para elegir (sin mirar el test):
    uv run python scripts/ejecutar.py exp30 --conjunto train --sin-figuras
    uv run python scripts/resumen.py --conjunto train --filtro exp30
La mejor configuración se copia a exp31_otsu_preprocesado.py.
"""

CANALES = [("gris", False), ("Cr", False), ("a", False), ("b", False), ("S", False), ("H", True)]
VARIANTES = ["clasico", "multi", "iterado"]

EXPERIMENTOS = [
    {
        "nombre": f"exp30_otsu_{variante}_{canal}",
        "descripcion": f"Otsu {variante} sobre el canal {canal}",
        "preprocesado": [],
        "modelo": ("otsu", {"canal": canal, "variante": variante, "invertir": invertir}),
    }
    for variante in VARIANTES
    for canal, invertir in CANALES
]

EXPERIMENTOS += [
    {
        "nombre": f"exp30_otsu_{variante}_Cr_{operacion}_b",
        "descripcion": f"Otsu {variante} en Cr y b*, combinados con {operacion.upper()}",
        "preprocesado": [],
        "modelo": ("otsu_combinado", {"canales": ("Cr", "b"), "operacion": operacion, "variante": variante}),
    }
    for variante in VARIANTES
    for operacion in ("and", "or")
]
