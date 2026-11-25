import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras

from consultas_salud import biblioteca, consultar_con_ruta, es_consulta_personal
from consultas_proyecto import consultar_proyecto
from vistas_radiografia import preparar_vistas

RAIZ = Path(__file__).resolve().parent
RECURSOS = RAIZ / "recursos"
ENCUADRE = RAIZ / "experimento_encuadre"
BALANCE = RAIZ / "experimento_balance_profundidad" / "resultado_s42"
VALIDACION = RAIZ / "experimento_umbral_validacion"
NOMBRES_CNN = {
    "original": "CNN original",
    "cuatro_vistas": "CNN de cuatro vistas",
    "profunda42": "CNN profunda con lotes 50/50",
}

st.set_page_config(page_title="Radiografías de tórax", page_icon="🩻")
st.title("Radiografías de tórax")
st.caption("Prototipo académico · Tres CNN entrenadas desde cero con radiografías pediátricas. Las tres deben votar PNEUMONIA para asignar esa clase.")


def leer_json(ruta):
    return json.loads(ruta.read_text(encoding="utf-8"))


@st.cache_resource
def cargar_redes():
    from comparar_encuadre import cargar_candidato

    return {
        "original": keras.models.load_model(RECURSOS / "mejor.keras", compile=False),
        "cuatro_vistas": cargar_candidato(ENCUADRE / "candidato.keras"),
        "profunda42": keras.models.load_model(BALANCE / "propuesta.keras", compile=False),
    }


def preparar(archivo):
    with Image.open(archivo) as original:
        gris = original.convert("L")
    escala = 160 / max(gris.size)
    medidas = tuple(max(1, round(d * escala)) for d in gris.size)
    ajustada = gris.resize(medidas, Image.Resampling.LANCZOS)
    cuadro = Image.new("L", (160, 160))
    cuadro.paste(ajustada, ((160 - medidas[0]) // 2, (160 - medidas[1]) // 2))
    return cuadro


def informacion():
    cortes = leer_json(VALIDACION / "regla_validacion_completa_congelada.json")["cortes"]
    verificacion = leer_json(VALIDACION / "verificacion_desde_jpeg.json")
    metricas = verificacion["matriz_nueva"]
    metricas["exactitud"] = (metricas["VP"] + metricas["VN"]) / verificacion["n"]
    resumen_cortes = ", ".join(f"{nombre} {cortes[clave]:.6f}" for clave, nombre in NOMBRES_CNN.items())
    regla = ("Para asignar PNEUMONIA, las tres CNN deben alcanzar o superar su propio umbral. "
             "Si alguna queda por debajo, la salida es NORMAL. Umbrales aproximados: "
             f"{resumen_cortes}. La clasificación utiliza los valores completos guardados en validación.")
    entrenamiento = ("Las tres CNN se entrenaron desde cero con el dataset pediátrico del profesor. "
                     "La CNN original usó 4.185 imágenes, la de cuatro vistas usó 3.788 y la CNN profunda "
                     "usó 3.788 con lotes equilibrados, mitad NORMAL y mitad PNEUMONIA.")
    return metricas, cortes, regla, entrenamiento


def clasificar(cuadro, cortes):
    redes = cargar_redes()
    gris = np.asarray(cuadro, dtype=np.float32)[None, :, :, None] / 255
    vistas = preparar_vistas(cuadro)[None].astype(np.float32) / 255
    # Replicar la imagen mantiene la forma del lote de calibración: un redondeo
    # de la inferencia unitaria puede invertir un voto exactamente en el corte.
    lote_gris = np.repeat(gris, 32, axis=0)
    lote_vistas = np.repeat(vistas, 32, axis=0)
    entradas = {"original": lote_gris, "cuatro_vistas": lote_vistas, "profunda42": lote_gris}
    votos, detalle = [], []
    for clave, nombre in NOMBRES_CNN.items():
        puntaje = float(redes[clave](entradas[clave], training=False).numpy()[0, 0])
        voto = puntaje >= cortes[clave]
        votos.append(voto)
        detalle.append(f"{nombre} · {'PNEUMONIA' if voto else 'NORMAL'} · "
                       f"puntaje {puntaje:.6f} · umbral {cortes[clave]:.6f}")
    return all(votos), "\n\n".join(detalle)


metricas, cortes, regla, entrenamiento = informacion()
st.download_button("Descargar radiografía de ejemplo (.jpg)", (RECURSOS / "ejemplo.jpeg").read_bytes(), file_name="radiografia_ejemplo.jpg", mime="image/jpeg")
archivo = st.file_uploader("Carga una radiografía de pecho", type=["jpg", "jpeg", "png"])
if archivo is None and st.checkbox("Usar una radiografía de ejemplo del dataset"):
    archivo = RECURSOS / "ejemplo.jpeg"
resultado = None
if archivo is not None:
    try:
        vista = preparar(archivo)
    except (OSError, ValueError, Image.DecompressionBombError):
        st.error("No se pudo leer la imagen. Usa un archivo JPEG o PNG válido.")
    else:
        positivo, detalle = clasificar(vista, cortes)
        clase = "PNEUMONIA" if positivo else "NORMAL"
        resultado = f"El modelo de tres CNN clasificó esta imagen como {clase}. {detalle}."
        izquierda, derecha = st.columns(2)
        izquierda.image(vista, caption="Radiografía preparada · 160 × 160")
        derecha.metric("Clasificación del modelo", clase)
        derecha.write(detalle)
st.subheader("Pregúntale al asistente")
salud, proyecto = st.tabs(["Sobre la neumonía", "Sobre la imagen y el modelo"])


def mostrar_ficha(ficha):
    fuente = ficha["fuente"]
    st.markdown(f"**{ficha['titulo']}**\n\n{ficha['texto']}\n\n"
                f"[{fuente['entidad']} · {fuente['titulo']}]({fuente['url']})")


def mostrar_consulta(consulta):
    st.text(consulta["pregunta"])
    if consulta["fichas"]:
        for ficha in consulta["fichas"]:
            mostrar_ficha(ficha)
        explicaciones = {
            "acuerdo": "El Transformer local y la búsqueda por términos coincidieron en la ficha.",
            "desempate": "El Transformer local eligió entre fichas con términos similares.",
            "lexico": "La ficha se seleccionó por coincidencia de términos con la pregunta.",
            "urgencia": "Se mostró la ficha de urgencias por las señales descritas.",
            "edades": "Se añadieron las fichas de los grupos de edad consultados.",
        }
        for ruta in dict.fromkeys(consulta.get("rutas", [])):
            st.caption(explicaciones[ruta])
    else:
        st.info(consulta["aviso"])


with salud:
    historial = st.session_state.setdefault("consultas_de_salud", [])
    with st.form("pregunta_de_salud", clear_on_submit=True):
        pregunta_salud = st.text_input("Tu pregunta sobre neumonía", max_chars=700,
                                       placeholder="¿Qué síntomas presenta la neumonía en bebés?")
        enviar = st.form_submit_button("Preguntar", type="primary")
    if enviar and pregunta_salud.strip():
        tema = historial[-1]["tema"] if historial else None
        fichas, rutas = consultar_con_ruta(pregunta_salud, tema_anterior=tema)
        aviso = ""
        if not fichas:
            if es_consulta_personal(pregunta_salud):
                aviso = ("Un profesional de salud debe valorar el caso para indicar un tratamiento o una dosis. "
                         "Puedo explicar qué se conoce en general y mostrarte la fuente.")
            else:
                aviso = ("Las fuentes disponibles no me permiten contestar esa pregunta con confianza. "
                         "Puedes precisar el tema o elegirlo en «Explorar temas y fuentes».")
        historial.append(dict(pregunta=pregunta_salud.strip(), fichas=fichas, rutas=rutas, aviso=aviso,
                              tema=fichas[0]["id"] if len(fichas) == 1 else None))
        del historial[:-8]
    if historial:
        mostrar_consulta(historial[-1])
        if len(historial) > 1:
            with st.expander("Preguntas anteriores de esta sesión"):
                for consulta in reversed(historial[:-1]):
                    mostrar_consulta(consulta)
    with st.expander("Explorar temas y fuentes"):
        fichas_indexadas, _ = biblioteca()
        por_titulo = {f["titulo"]: f for f, _, _ in fichas_indexadas}
        tema_elegido = st.selectbox("Tema", list(por_titulo), index=None, placeholder="Selecciona un tema")
        if tema_elegido is not None:
            mostrar_ficha(por_titulo[tema_elegido])
        else:
            fuentes = {f["fuente"]["url"]: f["fuente"] for f in por_titulo.values()}
            for enlace, fuente in fuentes.items():
                st.markdown(f"[{fuente['entidad']} · {fuente['titulo']}]({enlace})")

with proyecto:
    with st.form("pregunta_de_proyecto", clear_on_submit=True):
        pregunta = st.text_input("Tu pregunta sobre la imagen o el modelo", max_chars=700,
                                 placeholder="¿Qué predijo en esta radiografía? ¿Cómo se calcula el F1?")
        enviar_proyecto = st.form_submit_button("Preguntar sobre el proyecto", type="primary")
    if enviar_proyecto and pregunta.strip():
        anterior = st.session_state.get("ultima_pregunta_proyecto", "")
        st.session_state["pregunta_previa_proyecto"] = anterior
        st.session_state["ultima_pregunta_proyecto"] = pregunta.strip()
    actual = st.session_state.get("ultima_pregunta_proyecto")
    if actual:
        st.text(actual)
        st.markdown(consultar_proyecto(actual, metricas, regla, entrenamiento, resultado,
                                      pregunta_anterior=st.session_state.get("pregunta_previa_proyecto", "")))
