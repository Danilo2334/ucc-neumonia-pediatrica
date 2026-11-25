"""Cuatro vistas en escala de grises a partir del cuadro de 160 × 160."""

import numpy as np
from PIL import Image


def preparar_vistas(cuadro):
    gris = cuadro.convert("L")
    contenido = gris.crop(gris.getbbox() or (0, 0, *gris.size))
    ancho, alto = contenido.size
    lado = min(ancho, alto)
    vistas = [gris]
    for posicion in (0, 0.5, 1):
        x = round((ancho - lado) * posicion)
        y = round((alto - lado) * posicion)
        vistas.append(contenido.crop((x, y, x + lado, y + lado)))
    return np.stack([np.asarray(v.resize((112, 112), Image.Resampling.LANCZOS))
                     for v in vistas], axis=-1)
