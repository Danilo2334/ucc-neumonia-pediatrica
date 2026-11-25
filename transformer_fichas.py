"""Encoder Transformer local para ordenar las fichas educativas por pregunta."""
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np


RECURSOS = Path(__file__).resolve().parent / "recursos"
CORPUS = RECURSOS / "fuentes_neumonia.json"
DIRECTORIO = RECURSOS / "transformer_fichas"
FORMATO = 1
LONGITUD = 28
PIEZAS = 32
DIMENSION = 40


def normalizar_texto(texto):
    letras = unicodedata.normalize("NFKD", texto.casefold())
    return " ".join(re.findall(r"[a-z0-9]+", "".join(c for c in letras if not unicodedata.combining(c))))


def fragmentos(palabra):
    # Compartir fragmentos permite relacionar flexiones sin embeddings externos.
    delimitada = "^" + palabra + "$"
    partes = [delimitada[i:i + n] for n in (3, 4) for i in range(len(delimitada) - n + 1)]
    return list(dict.fromkeys(["=" + palabra, *partes]))[:PIEZAS]


def vocabulario_de(textos):
    frecuencia, piezas = Counter(), set()
    for texto in textos:
        palabras = set(normalizar_texto(texto).split())
        frecuencia.update(palabras)
        for palabra in palabras:
            piezas.update(fragmentos(palabra))
    ids = {f: i for i, f in enumerate(sorted(piezas), 1)}
    importancia = {p: 1 + math.log((1 + len(textos)) / (1 + n)) for p, n in frecuencia.items()}
    return ids, importancia


def entradas(textos, vocabulario, idf):
    tokens = np.zeros((len(textos), LONGITUD, PIEZAS), dtype=np.int32)
    pesos = np.zeros((len(textos), LONGITUD), dtype=np.float32)
    for fila, texto in enumerate(textos):
        for col, palabra in enumerate(normalizar_texto(texto).split()[:LONGITUD]):
            partes = fragmentos(palabra)
            conocidos = [vocabulario[f] for f in partes if f in vocabulario]
            if conocidos:
                tokens[fila, col, :len(conocidos)] = conocidos
                # Una coincidencia subléxica incidental no vale tanto como una palabra vista.
                pesos[fila, col] = idf.get(palabra, 3.) * (len(conocidos) / len(partes))
    return {"fragmentos": tokens, "importancia": pesos}


def construir_modelo(n_vocabulario, n_fichas):
    # La importación diferida permite usar la biblioteca y sus reglas sin cargar TensorFlow.
    import keras
    from keras import layers, ops

    piezas = keras.Input((LONGITUD, PIEZAS), dtype="int32", name="fragmentos")
    importancia = keras.Input((LONGITUD,), dtype="float32", name="importancia")
    emb = layers.Embedding(n_vocabulario + 1, DIMENSION, name="embedding_desde_cero")(piezas)
    presentes = ops.cast(ops.not_equal(piezas, 0), "float32")
    total = ops.sum(emb * ops.expand_dims(presentes, -1), axis=2)
    cuenta = ops.maximum(ops.sum(presentes, axis=2, keepdims=True), 1.)
    palabras = layers.LayerNormalization(name="escala_palabras")(total / ops.sqrt(cuenta))
    # Posición aprendida en el mismo entrenamiento que el resto del encoder.
    posiciones = ops.cumsum(ops.ones_like(importancia, dtype="int32"), axis=1) - 1
    pos = layers.Embedding(LONGITUD, DIMENSION, name="posicion")(posiciones)
    x = layers.Add()([palabras, pos])
    mascara = ops.expand_dims(ops.greater(importancia, 0.), 1)
    atendido = layers.MultiHeadAttention(num_heads=4, key_dim=DIMENSION // 4, dropout=.08,
                                        name="autoatencion")(x, x, attention_mask=mascara)
    x = layers.LayerNormalization(name="residuo_atencion")(layers.Add()([x, atendido]))
    ff = layers.Dense(2 * DIMENSION, activation="gelu", name="feedforward")(x)
    ff = layers.Dense(DIMENSION)(ff)
    x = layers.LayerNormalization(name="residuo_feedforward")(layers.Add()([x, ff]))
    pond = ops.expand_dims(importancia, -1)
    vector = ops.sum(x * pond, axis=1) / ops.maximum(ops.sum(pond, axis=1), 1.)
    vector = layers.Dropout(.12)(vector)
    salida = layers.Dense(n_fichas, activation="softmax", name="ficha")(vector)
    return keras.Model([piezas, importancia], salida, name="transformer_fichas")


def huella(archivo):
    return hashlib.sha256(Path(archivo).read_bytes()).hexdigest()


def _sello(ruta):
    datos = ruta.stat()
    return datos.st_mtime_ns, datos.st_size


@lru_cache(maxsize=2)
def _leer_meta(ruta, sello):
    return json.loads(ruta.read_text(encoding="utf-8"))


@lru_cache(maxsize=2)
def _cargar(corpus, sello_corpus, archivo, sello_meta, pesos, sello_pesos):
    meta = _leer_meta(archivo, sello_meta)
    if meta["formato"] != FORMATO or huella(corpus) != meta["corpus_sha256"]:
        raise ValueError("Las fichas o el formato cambiaron. Se requiere volver a entrenar.")
    if huella(pesos) != meta["pesos_sha256"]:
        raise ValueError("Los pesos del Transformer no coinciden con su registro.")
    modelo = construir_modelo(len(meta["vocabulario"]), len(meta["clases"]))
    modelo.load_weights(pesos)
    return meta, modelo


def _recursos(archivo_corpus):
    corpus = Path(archivo_corpus or CORPUS).resolve()
    archivo = DIRECTORIO / "modelo.json"
    sello_meta = _sello(archivo)
    meta = _leer_meta(archivo, sello_meta)
    pesos = DIRECTORIO / Path(meta["pesos"]).name
    return _cargar(corpus, _sello(corpus), archivo, sello_meta, pesos, _sello(pesos))


def estado_modelo(archivo_corpus=None):
    try:
        meta, _ = _recursos(archivo_corpus)
        return {"disponible": True, "motivo": "", **meta["criterio_confianza"]}
    except (OSError, ValueError, TypeError, KeyError, ImportError) as error:
        return {"disponible": False, "motivo": str(error)}


def ranquear(texto, archivo_corpus=None):
    """Puntajes softmax relativos entre fichas. Una lista vacía indica falta de cobertura o recursos."""
    try:
        meta, modelo = _recursos(archivo_corpus)
    except (OSError, ValueError, TypeError, KeyError, ImportError):
        return []
    x = entradas([texto], meta["vocabulario"], meta["idf"])
    if not np.any(x["importancia"]):
        return []
    probabilidades = np.asarray(modelo(x, training=False))[0]
    orden = np.argsort(-probabilidades, kind="stable")
    return [(meta["clases"][i], float(probabilidades[i])) for i in orden]
