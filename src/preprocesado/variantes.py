"""Preprocesados comunes que prueba cada método (elegidos con scripts/analizar_preprocesado.py)."""

BILATERAL = ("bilateral", {"d": 9, "sigma_color": 75, "sigma_espacio": 75})

PREPROCESADOS = {
    "P0_sin": [],
    "P1_gaussiano": [("gaussiano", {"k": 5})],
    "P2_bilateral": [BILATERAL],
    "P3_bilateral_clahe": [BILATERAL, ("clahe", {"clip": 2.0, "rejilla": 8})],
}

OPCIONALES = {
    "gamma_07": [("gamma", {"valor": 0.7})],            # solo métodos que usen el canal S
    "roi_085": [("roi", {"y_fin": 0.85})],              # solo en el pipeline final
}