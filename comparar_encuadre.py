"""Entrenar una CNN desde cero, fijarla con validación y después compararla en prueba."""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow import keras

from comparar_peso_normal import leer, particionar
from vistas_radiografia import preparar_vistas


def guardar(ruta, datos):
    ruta.write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")


def codigo(ruta):
    patron = (r"(person[0-9]+)_.+" if ruta.parent.name == "PNEUMONIA"
              else r"((?:NORMAL2-)?IM-[0-9]+)(?:-[0-9]+)+")
    return re.fullmatch(patron, ruta.stem, flags=re.I)[1].upper()


def filas_prueba(raiz):
    return [(ruta, etiqueta) for etiqueta, clase in enumerate(("NORMAL", "PNEUMONIA"))
            for ruta in sorted((raiz / "test" / clase).iterdir())
            if ruta.suffix.lower() in {".png", ".jpg", ".jpeg"} and not ruta.name.startswith(".")]


def huella(ruta):
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def separar(raiz):
    train, val = particionar(raiz)
    test = filas_prueba(raiz)
    codigos_test = {codigo(ruta) for ruta, _ in test}
    comunes = codigos_test & {codigo(ruta) for ruta, _ in train + val}
    # La huella también encuentra copias byte a byte con otro nombre.
    tamanos = {ruta.stat().st_size for ruta, _ in test}
    huellas = {huella(ruta) for ruta, _ in test}
    copias = {codigo(ruta) for ruta, _ in train + val
              if ruta.stat().st_size in tamanos and huella(ruta) in huellas}
    excluidos = comunes | copias
    filtrados = [[fila for fila in grupo if codigo(fila[0]) not in excluidos] for grupo in (train, val)]
    cod_train, cod_val = [{codigo(ruta) for ruta, _ in grupo} for grupo in filtrados]
    assert not (cod_train & cod_val or cod_train & codigos_test or cod_val & codigos_test)
    detalle = dict(codigos_coincidentes=len(comunes), codigos_con_copia_exacta=len(copias),
                   imagenes_excluidas=len(train) + len(val) - sum(map(len, filtrados)),
                   train=np.bincount([y for _, y in filtrados[0]]).tolist(),
                   val=np.bincount([y for _, y in filtrados[1]]).tolist())
    return filtrados[0], filtrados[1], detalle


def ancho_alto(ruta):
    with Image.open(ruta) as im:
        return im.width / im.height


def pesos_entrenamiento(filas, y):
    proporcion = np.array([ancho_alto(ruta) for ruta, _ in filas])
    cortes = np.quantile(proporcion[y == 0], [1/3, 2/3])
    grupo = 3 * y + np.digitize(proporcion, cortes)
    frecuencias = np.bincount(grupo, minlength=6)
    assert np.all(frecuencias > 0)
    pesos = np.minimum(len(y) / (6 * frecuencias[grupo]), 3.0)
    return (pesos / pesos.mean()).astype("float32"), cortes.tolist()


def crear_red(pasos):
    capas = keras.layers
    entrada = keras.Input((112, 112, 4), name="cuatro_vistas_grises")
    x = capas.RandomTranslation(0.035, 0.035, fill_mode="nearest", seed=42)(entrada)
    x = capas.RandomContrast(0.12, value_range=(0, 1), seed=43)(x)
    x = capas.LayerNormalization(axis=(1, 2), scale=False, center=False, epsilon=0.001)(x)
    for i, filtros in enumerate((16, 24, 40, 64)):
        x = capas.Conv2D(filtros, 5 if i == 0 else 3, strides=2 if i == 0 else 1,
                         padding="same", use_bias=False)(x)
        x = capas.BatchNormalization(momentum=0.9)(x)
        x = capas.Activation("relu")(x)
        x = capas.MaxPooling2D()(x)
    x = capas.Flatten()(x)
    x = capas.Dense(48, activation="relu")(x)
    x = capas.Dropout(0.25)(x)
    salida = capas.Dense(1, activation="sigmoid")(x)
    red = keras.Model(entrada, salida, name="cnn_cuatro_vistas")
    tasa = keras.optimizers.schedules.CosineDecay(0.0008, pasos * 12, alpha=0.1)
    red.compile(optimizer=keras.optimizers.Adam(tasa), loss="binary_crossentropy")
    return red


def vistas(cuadros):
    return np.stack([preparar_vistas(Image.fromarray(x[:, :, 0])) for x in cuadros])


def cargar_candidato(archivo):
    # Reconstruir la misma arquitectura permite usar los pesos en distintas versiones de Keras.
    red = crear_red(1)
    red.load_weights(archivo)
    return red


def puntuar(red, x):
    return np.concatenate([red(x[i:i+32].astype("float32") / 255, training=False).numpy().ravel()
                           for i in range(0, len(x), 32)])


def metricas(y, p, umbral):
    vn, fp, fn, vp = np.bincount(2 * y + (p >= umbral), minlength=4).tolist()
    normal, neumonia = vn / (vn + fp), vp / (vp + fn)
    return dict(VN=vn, FP=fp, FN=fn, VP=vp, recall_normal=normal, recall_neumonia=neumonia,
                exactitud=(vn + vp) / len(y), balanceada=(normal + neumonia) / 2)


def entrenar(raiz, destino):
    destino.mkdir(parents=True, exist_ok=False)
    keras.utils.set_random_seed(42)
    train, val, detalle = separar(raiz)
    print("Partición depurada", detalle, flush=True)
    cuadros, y = leer(train)
    x = vistas(cuadros)
    del cuadros
    cuadros_val, y_val = leer(val)
    x_val = vistas(cuadros_val)
    del cuadros_val
    pesos, cortes = pesos_entrenamiento(train, y)
    red = crear_red(math.ceil(len(train) / 32))
    azar = np.random.default_rng(42)
    historial = []
    mejor = float("inf")
    archivo = destino / "candidato.keras"
    for epoca in range(1, 13):
        orden = azar.permutation(len(y))
        for i in range(0, len(y), 32):
            ids = orden[i:i+32]
            red.train_on_batch(x[ids].astype("float32") / 255, y[ids, None],
                               sample_weight=pesos[ids])
        p = np.clip(puntuar(red, x_val), 1e-7, 1 - 1e-7)
        perdida = float((-np.log1p(-p[y_val == 0]).mean() - np.log(p[y_val == 1]).mean()) / 2)
        historial.append(dict(epoca=epoca, perdida_balanceada_val=perdida))
        print(f"Época {epoca:2} de 12 | pérdida balanceada de validación {perdida:.4f}", flush=True)
        if perdida < mejor:
            mejor, mejor_epoca = perdida, epoca
            red.save(archivo)
        if epoca - mejor_epoca >= 3:
            break
    red = keras.models.load_model(archivo, compile=False)
    p = puntuar(red, x_val)
    opciones = [(t/100, metricas(y_val, p, t/100)) for t in range(0, 101)]
    admisibles = [(t, m) for t, m in opciones if m["recall_neumonia"] >= 0.94]
    umbral, validacion = max(admisibles, key=lambda z: (z[1]["VN"], z[1]["VP"], -abs(z[0] - 0.5)))
    registro = dict(particion=detalle, cortes_proporcion_train=cortes, historial=historial,
                    epoca=mejor_epoca, umbral=umbral, sensibilidad_minima_val=0.94,
                    validacion=validacion, sha256_modelo=huella(archivo))
    guardar(destino / "seleccion_validacion.json", registro)
    print("Selección cerrada en validación", mejor_epoca, umbral, validacion, flush=True)
    return registro


def evaluar_prueba(raiz, destino, referencia):
    seleccion = json.loads((destino / "seleccion_validacion.json").read_text(encoding="utf-8"))
    archivo = destino / "candidato.keras"
    assert huella(archivo) == seleccion["sha256_modelo"]
    test = filas_prueba(raiz)
    cuadros, y = leer(test)
    assert np.bincount(y).tolist() == [234, 390]
    base = keras.models.load_model(referencia, compile=False)
    nueva = cargar_candidato(archivo)
    anterior = puntuar(base, cuadros)
    actual = puntuar(nueva, vistas(cuadros))
    previo = metricas(y, anterior, 0.71)
    assert [previo[k] for k in ("VN", "FP", "FN", "VP")] == [142, 92, 22, 368]
    umbral = seleccion["umbral"]
    posterior = metricas(y, actual, umbral)
    detalle = {}
    for c, clase in enumerate(("NORMAL", "PNEUMONIA")):
        a = (anterior >= 0.71) == c
        b = (actual >= umbral) == c
        detalle[clase] = dict(errores_corregidos=int(np.sum((y == c) & ~a & b)),
                              aciertos_perdidos=int(np.sum((y == c) & a & ~b)))
    resultado = dict(original=previo, cuatro_vistas=posterior, umbral=umbral,
                     cambios_por_imagen=detalle, evaluacion="comparación en el test histórico")
    guardar(destino / "comparacion_test.json", resultado)
    print(json.dumps(resultado, indent=2), flush=True)
    return resultado


if __name__ == "__main__":
    import tensorflow as tf
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("etapa", choices=("entrenar", "evaluar"))
    parser.add_argument("raiz", type=Path)
    parser.add_argument("destino", type=Path)
    parser.add_argument("--referencia", type=Path, default=Path(__file__).parent / "recursos/mejor.keras")
    args = parser.parse_args()
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism()
    if args.etapa == "entrenar":
        entrenar(args.raiz, args.destino)
    else:
        evaluar_prueba(args.raiz, args.destino, args.referencia)
