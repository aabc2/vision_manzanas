"""Crecimiento de regiones, fase 1: criterio de color, tolerancia, criterio geométrico y semillas.

Sin preprocesado ni postprocesado, para comparar solo los parámetros del método:
    - referencia: solo las semillas (iteraciones=0), que equivale a Otsu iterado Cr AND b*.
    - color H (tono, distancia circular) con tolerancias 2, 3, 5 y 8, sin y con criterio
      geométrico (círculo envolvente de la semilla x1.0 y x1.15).
    - color a*b* y Cr/b* con tolerancias 6, 10 y 15, con criterio geométrico x1.0.
    - semillas por percentil (sin Otsu) para ver cuánto dependen las semillas.

Se ejecuta sobre TRAIN para elegir (sin mirar el test):
    uv run python scripts/ejecutar.py exp40 --conjunto train --sin-figuras
    uv run python scripts/resumen.py --conjunto train --filtro exp40
Las dos mejores configuraciones se copian a exp41_regiones_preprocesado.py.
"""

GEOMETRIA = {"sin_geo": {"geometrico": False}, "geo100": {"geometrico": True, "factor": 1.0},
             "geo115": {"geometrico": True, "factor": 1.15}}

EXPERIMENTOS = [
    {
        "nombre": "exp40_reg_solo_semillas",
        "descripcion": "Referencia: solo las semillas (Otsu iterado Cr AND b*), sin crecer",
        "preprocesado": [],
        "modelo": ("crecimiento", {"iteraciones": 0}),
    }
]

EXPERIMENTOS += [
    {
        "nombre": f"exp40_reg_H_t{tol}_{id_geo}",
        "descripcion": f"Crecimiento por tono H, tolerancia {tol}, {id_geo}",
        "preprocesado": [],
        "modelo": ("crecimiento", {"color": "H", "tolerancia": tol, "iteraciones": 25, **geo}),
    }
    for tol in (2, 3, 5, 8)
    for id_geo, geo in GEOMETRIA.items()
]

EXPERIMENTOS += [
    {
        "nombre": f"exp40_reg_{color}_t{tol}_geo100",
        "descripcion": f"Crecimiento por {color}, tolerancia {tol}, con criterio geométrico",
        "preprocesado": [],
        "modelo": ("crecimiento", {"color": color, "tolerancia": tol, "iteraciones": 25, "geometrico": True, "factor": 1.0}),
    }
    for color in ("ab", "Crb")
    for tol in (6, 10, 15)
]

EXPERIMENTOS += [
    {
        "nombre": f"exp40_reg_H_t3_geo100_semillas_p{p}",
        "descripcion": f"Como H t3 geo100, pero con semillas del {p} % superior de Cr y b* (sin Otsu)",
        "preprocesado": [],
        "modelo": ("crecimiento", {"color": "H", "tolerancia": 3, "iteraciones": 25, "geometrico": True,
                                    "factor": 1.0, "semillas": "percentil", "percentil": p}),
    }
    for p in (2, 4)
]
