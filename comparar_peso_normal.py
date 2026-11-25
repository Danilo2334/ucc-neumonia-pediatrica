"""Compara una época adicional, con y sin ponderación, usando solo train y val."""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras


def particionar(raiz):
    azar = random.Random(42)
    train, val = [], []
    for etiqueta, clase in enumerate(("NORMAL", "PNEUMONIA")):
        grupos = defaultdict(list)
        for carpeta in ("train", "val"):
            for ruta in sorted((raiz / carpeta / clase).iterdir()):
                if ruta.name.startswith(".") or ruta.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                patron = r"(person[0-9]+)_.+" if etiqueta else r"((?:NORMAL2-)?IM-[0-9]+)(?:-[0-9]+)+"
                codigo = re.fullmatch(patron, ruta.stem, re.I)
                if codigo is None:
                    raise ValueError(f"Nombre sin código reconocido: {ruta.name}")
                grupos[codigo[1].upper()].append((ruta, etiqueta))
        codigos = sorted(grupos)
        azar.shuffle(codigos)
        objetivo = round(sum(map(len, grupos.values())) * 0.20)
        en_val = 0
        for codigo in codigos:
            grupo = grupos[codigo]
            if en_val + len(grupo) <= objetivo:
                val.extend(grupo)
                en_val += len(grupo)
            else:
                train.extend(grupo)
        assert en_val == objetivo
    assert (len(train), len(val)) == (4185, 1047)
    return train, val


def leer(filas):
    imagenes = np.empty((len(filas), 160, 160, 1), dtype=np.uint8)
    for i, (ruta, _) in enumerate(filas):
        with Image.open(ruta) as original:
            gris = original.convert("L")
        escala = 160 / max(gris.size)
        medidas = tuple(max(1, round(d * escala)) for d in gris.size)
        cuadro = Image.new("L", (160, 160))
        cuadro.paste(gris.resize(medidas, Image.Resampling.LANCZOS),
                     ((160 - medidas[0]) // 2, (160 - medidas[1]) // 2))
        imagenes[i, :, :, 0] = np.asarray(cuadro)
    return imagenes, np.array([etiqueta for _, etiqueta in filas], dtype=np.int32)


def evaluar(modelo, x, y):
    puntajes = np.concatenate([
        modelo(x[i:i+32].astype(np.float32) / 255, training=False).numpy().ravel()
        for i in range(0, len(y), 32)
    ])
    pred = puntajes >= 0.71
    vn, fp, fn, vp = np.bincount(2*y + pred, minlength=4).tolist()
    recall_normal = vn / (vn + fp)
    recall_neumonia = vp / (vp + fn)
    return dict(VN=vn, FP=fp, FN=fn, VP=vp, recall_normal=recall_normal,
                recall_neumonia=recall_neumonia, balanceada=(recall_normal + recall_neumonia)/2)


def ejecutar(raiz, destino):
    keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    destino.mkdir(parents=True, exist_ok=False)
    train, val = particionar(raiz)
    recursos = Path(__file__).resolve().parent / "recursos"
    ruta_modelo = recursos / "mejor.keras"
    x_val, y_val = leer(val)
    base = keras.models.load_model(ruta_modelo, compile=False)
    resultados = {"epoca_10": evaluar(base, x_val, y_val)}
    referencia = json.loads((recursos / "umbral.json").read_text())
    assert np.isclose(resultados["epoca_10"]["balanceada"],
                      referencia["exactitud_balanceada_val"], atol=1e-10, rtol=0)
    print("Validación de referencia:", resultados, flush=True)
    x, y = leer(train)
    orden = np.random.default_rng(42).permutation(len(y))
    for nombre, peso in [("control", 1.0), ("normal_1_5", 1.5)]:
        modelo = keras.models.load_model(ruta_modelo, compile=False)
        modelo.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01),
                       loss="binary_crossentropy")
        for i in range(0, len(y), 32):
            ids = orden[i:i+32]
            pesos = np.where(y[ids] == 0, peso, 1.0).astype(np.float32)
            modelo.train_on_batch(x[ids].astype(np.float32) / 255, y[ids, None],
                                  sample_weight=pesos)
        resultados[nombre] = evaluar(modelo, x_val, y_val)
        print(nombre, resultados[nombre], flush=True)
        if nombre == "normal_1_5":
            modelo.save(destino / "candidato.keras")
    (destino / "validacion.json").write_text(json.dumps(resultados, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raiz", type=Path, help="Carpeta chest_xray con train y val.")
    parser.add_argument("destino", type=Path, help="Carpeta nueva para el experimento.")
    args = parser.parse_args()
    ejecutar(args.raiz, args.destino)
