"""Umbralización adaptativa, fase 1: canal, método, tamaño de bloque y C.

Sin preprocesado ni postprocesado, para comparar solo los parámetros del método:
    - línea base: canal gris (el peor canal según el análisis de Otsu).
    - canales: Cr (manzanas rojas), b* (amarillas) y las dos combinaciones
      Cr AND b* (más precisión) y Cr OR b* (más recall).
    - métodos: media y Gaussiana (cv.adaptiveThreshold).
    - bloque: 51, 101, 201 y 301 píxeles. Tiene que ser mayor que una manzana,
      o el centro de la manzana se compara consigo mismo y sale hueco.
    - C: -5, -10 y -20. Negativo: el píxel tiene que superar la media de su
      vecindad en |C| niveles; con C >= 0 se marcaría la mitad del fondo uniforme.
    - Niblack (umbral = media + k * desviación - C) en Cr AND b* y Cr OR b*,
      con C = -5 fijo y k = 0.2, 0.5 y 1.0.

Se ejecuta sobre TRAIN para elegir (sin mirar el test):
    uv run python scripts/ejecutar.py exp50 --conjunto train --sin-figuras
    uv run python scripts/resumen.py --conjunto train --filtro exp50
Las dos mejores configuraciones se copian a exp51_adaptativa_preprocesado.py.
"""

CANALES = {
    "Cr": ("adaptativa", {"canal": "Cr"}),
    "b": ("adaptativa", {"canal": "b"}),
    "Cr_and_b": ("adaptativa_combinada", {"canales": ("Cr", "b"), "operacion": "and"}),
    "Cr_or_b": ("adaptativa_combinada", {"canales": ("Cr", "b"), "operacion": "or"}),
}
METODOS = ["media", "gaussiana"]
BLOQUES = [51, 101, 201, 301]
CS = [-5, -10, -20]

# Línea base: gris, con un bloque y una C intermedios.
EXPERIMENTOS = [
    {
        "nombre": f"exp50_adapt_media_gris_b{bloque}_C-10",
        "descripcion": f"Línea base: adaptativa media sobre gris, bloque {bloque}, C = -10",
        "preprocesado": [],
        "modelo": ("adaptativa", {"canal": "gris", "metodo": "media", "bloque": bloque, "C": -10}),
    }
    for bloque in (101, 201)
]

# Media y Gaussiana en Cr, b* y sus combinaciones.
EXPERIMENTOS += [
    {
        "nombre": f"exp50_adapt_{metodo}_{id_canal}_b{bloque}_C{C}",
        "descripcion": f"Adaptativa {metodo} sobre {id_canal}, bloque {bloque}, C = {C}",
        "preprocesado": [],
        "modelo": (modelo, {**base, "metodo": metodo, "bloque": bloque, "C": C}),
    }
    for id_canal, (modelo, base) in CANALES.items()
    for metodo in METODOS
    for bloque in BLOQUES
    for C in CS
]

# Niblack en las dos combinaciones de canales.
EXPERIMENTOS += [
    {
        "nombre": f"exp50_adapt_niblack_{id_canal}_b{bloque}_k{k}",
        "descripcion": f"Niblack sobre {id_canal}, bloque {bloque}, k = {k}, C = -5",
        "preprocesado": [],
        "modelo": (modelo, {**base, "metodo": "niblack", "bloque": bloque, "k": k, "C": -5}),
    }
    for id_canal in ("Cr_and_b", "Cr_or_b")
    for modelo, base in [CANALES[id_canal]]
    for bloque in (101, 201, 301)
    for k in (0.2, 0.5, 1.0)
]

# Ampliación tras la primera ejecución: los mejores valores salieron en el borde
# de la rejilla (OR con C = -20, AND con bloque 301), y en AND faltaba el punto
# intermedio entre C = -10 (bueno) y C = -20 (muy malo).
AMPLIACION = (
    [("Cr_or_b", metodo, bloque, C) for metodo in METODOS for bloque in (101, 201, 301) for C in (-30, -40)]
    + [("Cr_and_b", metodo, bloque, -15) for metodo in METODOS for bloque in (201, 301, 401)]
    + [("Cr_and_b", metodo, 401, -10) for metodo in METODOS]
)
EXPERIMENTOS += [
    {
        "nombre": f"exp50_adapt_{metodo}_{id_canal}_b{bloque}_C{C}",
        "descripcion": f"Adaptativa {metodo} sobre {id_canal}, bloque {bloque}, C = {C} (ampliación)",
        "preprocesado": [],
        "modelo": (CANALES[id_canal][0], {**CANALES[id_canal][1], "metodo": metodo, "bloque": bloque, "C": C}),
    }
    for id_canal, metodo, bloque, C in AMPLIACION
]
EXPERIMENTOS.append({
    "nombre": "exp50_adapt_niblack_Cr_and_b_b401_k0.5",
    "descripcion": "Niblack sobre Cr_and_b, bloque 401, k = 0.5, C = -5 (ampliación)",
    "preprocesado": [],
    "modelo": ("adaptativa_combinada", {"canales": ("Cr", "b"), "operacion": "and",
                                        "metodo": "niblack", "bloque": 401, "k": 0.5, "C": -5}),
})