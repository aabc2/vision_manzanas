"""Preprocesado común a todos los métodos del proyecto.

Contrato: cada paso recibe una imagen BGR uint8 y devuelve otra BGR uint8 del
mismo tamaño. Los cambios de espacio de color (color.py) y de tamaño
(geometria.redimensionar) quedan fuera del pipeline a propósito.
"""

from .color import a_gris, canal, convertir
from .geometria import redimensionar
from .pipeline import PASOS, aplicar, describir

__all__ = ["PASOS", "aplicar", "describir", "canal", "convertir", "a_gris", "redimensionar"]
