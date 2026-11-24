"""K-means, fase 1: características, número de clústeres K y regla de selección.

Sin preprocesado ni postprocesado, para comparar solo los parámetros del método:
    - características de cada píxel:
        bgr   (sin cambio de espacio de color: línea base),
        lab   (los tres canales de Lab),
        ab    (Lab sin la luminosidad, para depender menos de la sombra),
        Cr_b  (los dos canales que mejor separan en Otsu).
    - K: 3, 4, 5, 6 y 8. La manzana ocupa ~5 % de la imagen: con K pequeño
      acaba en el mismo clúster que parte del fondo.
    - selección del clúster manzana (K-means no sabe cuál es):
        max_Cr          el clúster más rojo,
        max_suma        el de mayor Cr + b*,
        max_Cr_o_max_b  el más rojo y el más amarillo (rojas y amarillas),
        relativa        los que destacan sobre la media de la imagen en Cr o
                        b* por más de `margen` (10, 20 o 30), solo en ab y Cr_b.

Se ejecuta sobre TRAIN para elegir (sin mirar el test):
    uv run python scripts/ejecutar.py exp60 --conjunto train --sin-figuras
    uv run python scripts/resumen.py --conjunto train --filtro exp60
Las dos mejores configuraciones se copian a exp61_kmeans_preprocesado.py.
"""

CARACTERISTICAS = ["bgr", "lab", "ab", "Cr_b"]
KS = [3, 4, 5, 6, 8]
SELECCIONES = ["max_Cr", "max_suma", "max_Cr_o_max_b"]

EXPERIMENTOS = [
    {
        "nombre": f"exp60_kmeans_{car}_k{k}_{sel}",
        "descripcion": f"K-means sobre {car}, K = {k}, selección {sel}",
        "preprocesado": [],
        "modelo": ("kmeans", {"caracteristicas": car, "k": k, "seleccion": sel}),
    }
    for car in CARACTERISTICAS
    for k in KS
    for sel in SELECCIONES
]

EXPERIMENTOS += [
    {
        "nombre": f"exp60_kmeans_{car}_k{k}_relativa_m{margen}",
        "descripcion": f"K-means sobre {car}, K = {k}, selección relativa con margen {margen}",
        "preprocesado": [],
        "modelo": ("kmeans", {"caracteristicas": car, "k": k, "seleccion": "relativa", "margen": margen}),
    }
    for car in ("ab", "Cr_b")
    for k in (4, 6, 8)
    for margen in (10, 20, 30)
]

# Ampliación tras la primera ejecución (solo Cr_b, que fue claramente lo mejor):
#   - max_Cr / max_suma tienen el pico en K = 6, pero K = 7 no se había probado.
#   - relativa mejora al subir K y su mejor margen (30) era el mayor probado.
EXPERIMENTOS += [
    {
        "nombre": f"exp60_kmeans_Cr_b_k7_{sel}",
        "descripcion": f"K-means sobre Cr_b, K = 7, selección {sel} (ampliación)",
        "preprocesado": [],
        "modelo": ("kmeans", {"caracteristicas": "Cr_b", "k": 7, "seleccion": sel}),
    }
    for sel in ("max_Cr", "max_suma")
]
EXPERIMENTOS += [
    {
        "nombre": f"exp60_kmeans_Cr_b_k{k}_relativa_m{margen}",
        "descripcion": f"K-means sobre Cr_b, K = {k}, selección relativa con margen {margen} (ampliación)",
        "preprocesado": [],
        "modelo": ("kmeans", {"caracteristicas": "Cr_b", "k": k, "seleccion": "relativa", "margen": margen}),
    }
    for k, margen in [(6, 40), (6, 50), (8, 40), (8, 50),
                      (10, 30), (10, 40), (10, 50), (12, 30), (12, 40), (12, 50)]
]