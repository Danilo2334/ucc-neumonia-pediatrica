"""Tutor local del proyecto. Las métricas y la predicción llegan desde la app."""
import re
import unicodedata


def _normalizar(texto):
    texto = unicodedata.normalize("NFKD", texto.casefold())
    return " ".join(re.findall(r"[a-z0-9]+", "".join(c for c in texto if not unicodedata.combining(c))))


def _porcentaje(numerador, denominador):
    if not denominador:
        return "indefinido"
    return f"{100 * numerador / denominador:.2f}".replace(".", ",") + " %"


def _tabla_metricas(vp, fp, fn, vn):
    filas = ["| Clase | Precisión | Recall | F1 |", "|---|---:|---:|---:|"]
    for clase, aciertos, falsas, omitidas in (("NEUMONÍA", vp, fp, fn), ("NORMAL", vn, fn, fp)):
        valores = (_porcentaje(aciertos, aciertos + falsas),
                   _porcentaje(aciertos, aciertos + omitidas),
                   _porcentaje(2 * aciertos, 2 * aciertos + falsas + omitidas))
        filas.append(f"| {clase} | {' | '.join(valores)} |")
    return "\n".join(filas)


def _metricas(q, m):
    vp, fp, fn, vn = (m[k] for k in ("VP", "FP", "FN", "VN"))
    total = sum((vp, fp, fn, vn))
    normal = bool(re.search(r"\b(normal(?:es)?|negativ[ao]s?)\b", q))
    f1p, f1n = 2 * vp / (2 * vp + fp + fn), 2 * vn / (2 * vn + fp + fn)
    macro = f"F1 macro = (F1 de NEUMONÍA + F1 de NORMAL) / 2 = **{_porcentaje(f1p + f1n, 2)}**. Ambas clases tienen el mismo peso."
    if re.search(r"\bmatriz\b|tabla de confusion|cuadro de confusion", q):
        return ("Matriz del test histórico. Las filas son las predicciones y las columnas son las etiquetas reales.\n\n"
                "| Predicción | Real NEUMONÍA | Real NORMAL |\n|---|---:|---:|\n"
                f"| NEUMONÍA | VP = {vp} | FP = {fp} |\n| NORMAL | FN = {fn} | VN = {vn} |")
    if re.search(r"\b(fp|fals[ao]s? positiv[ao]s?|falsas? alarmas?|falsas? alertas?)\b|"
                 r"normal(?:es)?.{0,35}(clasific|marc|etiquet|confund|tom).{0,20}(neumonia|pneumonia)", q):
        if not re.search(r"negativ|\bfn\b", q):
            respuesta = (f"Hubo **{fp} falsos positivos**. De las {vn + fp} normales reales, {fp} recibieron "
                         f"la etiqueta PNEUMONIA y {vn} la etiqueta NORMAL.")
            if "por que" in q:
                respuesta += (" En esas imágenes las tres CNN superaron su umbral a pesar de la etiqueta NORMAL. "
                              "Los conteos no permiten identificar qué rasgo visual causó cada error.")
            return respuesta
    if re.search(r"\b(fn|fals[ao]s? negativ[ao]s?)\b|(neumonias?|neumonia clasificada).{0,45}como normal|"
                 r"(escapar|escaparon|omitio|omitieron|pasaron por alto).{0,20}neumon|neumonias?.{0,25}(escaparon|omitidas|perdidas)", q):
        if not re.search(r"fals[oa]s? positiv|\bfp\b", q):
            return (f"Hubo **{fn} falsos negativos**. De las {vp + fn} radiografías etiquetadas como "
                    f"PNEUMONIA, el sistema detectó {vp} y clasificó {fn} como NORMAL.")
    if re.search(r"\b(vp|verdaderos? positivos?)\b", q):
        return f"**{vp} verdaderos positivos.** Son las neumonías reales que el modelo también clasificó como PNEUMONIA."
    if re.search(r"\b(vn|verdaderos? negativos?)\b", q):
        return f"**{vn} verdaderos negativos.** Son las normales reales que el modelo también clasificó como NORMAL."
    if re.search(r"\b(error(?:es)?|fallos?|equivoco|equivocaciones|desaciertos?|falsos?|alarmas?)\b", q):
        return (f"En el test histórico hubo **{fp + fn} errores entre {total} radiografías**. "
                f"{fp} normales se clasificaron como PNEUMONIA (falsos positivos) y "
                f"{fn} neumonías como NORMAL (falsos negativos).")
    temas = {
        "exactitud": r"exactitud|accuracy|aciertos|acerto|acertaron",
        "precision": r"precision|valor predictivo",
        "sensibilidad": r"sensibilidad|detecta.{0,15}neumonia",
        "especificidad": r"especificidad",
        "recall": r"recall|exhaustividad",
        "f1": r"\bf\s?1\b|f score|media armonica|\bmacro\b",
    }
    elegidos = {tema for tema, patron in temas.items() if re.search(patron, q)}
    resumen = (f"En el test histórico se acertaron **{vp + vn} de {total}** "
               f"(exactitud **{_porcentaje(vp + vn, total)}**).")
    if len(elegidos) > 1 or re.search(r"metricas?|por clase|desempeno|rendimiento|classification report|"
                                        r"como (?:le )?(?:fue|rinde) (?:al |el )?modelo", q):
        return f"{resumen}\n\n{_tabla_metricas(vp, fp, fn, vn)}\n\n{macro}"
    if "exactitud" in elegidos:
        if normal:
            return f"De las {vn + fp} normales reales acertó {vn}. El recall de NORMAL es **{_porcentaje(vn, vn+fp)}**. " + resumen
        return resumen + f" Se calcula como (VP + VN) / total = ({vp} + {vn}) / {total}."
    if "precision" in elegidos:
        return (f"**Precisión** es la fracción de predicciones de una clase que fueron correctas. "
                f"Para NORMAL, VN / (VN + FN) = {vn}/{vn+fn} = **{_porcentaje(vn, vn+fn)}**. "
                f"Para NEUMONÍA, VP / (VP + FP) = {vp}/{vp+fp} = **{_porcentaje(vp, vp+fp)}**.")
    if "sensibilidad" in elegidos or "especificidad" in elegidos or "recall" in elegidos:
        return (f"**Recall de NEUMONÍA (sensibilidad)** = VP / (VP + FN) = {vp}/{vp+fn} = "
                f"**{_porcentaje(vp, vp+fn)}**.\n\n**Recall de NORMAL (especificidad)** = "
                f"VN / (VN + FP) = {vn}/{vn+fp} = **{_porcentaje(vn, vn+fp)}**. "
                "Cada denominador cuenta las imágenes reales de esa clase.")
    if "f1" in elegidos:
        return (f"El F1 combina precisión y recall con su media armónica: 2 × precisión × recall / (precisión + recall).\n\n"
                f"**NEUMONÍA:** 2 × VP / (2 × VP + FP + FN) = {2*vp}/{2*vp+fp+fn} = **{_porcentaje(f1p, 1)}**.\n\n"
                f"**NORMAL:** 2 × VN / (2 × VN + FP + FN) = {2*vn}/{2*vn+fp+fn} = **{_porcentaje(f1n, 1)}**.\n\n{macro}")
    return None


# Conceptos explicados con el preprocesamiento y las arquitecturas del proyecto.
LECCIONES = (
    (r"\bcuatro vistas\b|\b4 vistas\b|recortes|segund[oa] (cnn|red)",
     "La segunda CNN recibe cuatro vistas de la misma radiografía. Conservamos el cuadro completo y tomamos tres recortes cuadrados del contenido a lo largo de su eje mayor, al inicio, al centro y al final. Cada vista se lleva a 112 × 112 y se apilan como cuatro canales. Esa CNN emite un solo voto."),
    (r"agrup|mismo paciente|por paciente|codigo.{0,20}paciente|fuga de datos|solapamiento|data leakage|partici[oó]n|separaron|separamos",
     "Para separar entrenamiento y validación reunimos las imágenes que compartían código en el nombre de archivo. Así evitamos entrenar con una toma y evaluar con otra del mismo código. En la partición depurada también se excluyeron los códigos coincidentes con el test. El código es una aproximación al paciente, sin verificación mediante historias clínicas."),
    (r"aumento de datos|data augmentation|transformaciones aleatorias",
     "En el entrenamiento de la CNN de cuatro vistas se aplicaron desplazamientos aleatorios de hasta un 3,5 % y cambios de contraste. Al predecir en la app, esas transformaciones aleatorias están desactivadas."),
    (r"normaliza|\b255\b|divid.{0,15}pixel|escala de grises|intensidad|canal|alt[ou]ra.{0,15}ancho",
     "El cuadro de entrada mide 160 × 160 × 1: alto, ancho y un canal de gris. Cada intensidad se divide por 255. Por ejemplo, 128/255 ≈ 0,502. El valor 0 describe negro y 1 describe blanco. La etiqueta del archivo se guarda aparte: 0 para NORMAL y 1 para PNEUMONIA."),
    (r"tamano|dimensi[oó]n|160|106|1152|redimension|padding|relleno|borde|estira|preproces|"
     r"dar forma|forma.{0,18}entrada|entrada.{0,14}(cnn|red)|prepar.{0,16}(imagen|radiografia)",
     "La app convierte la imagen a grises, ajusta su lado mayor a 160 píxeles y conserva su proporción. Una imagen de 1152 × 760 queda en 160 × 106; se centra y se añaden 27 filas negras arriba y 27 abajo. El cuadro completo sí mide 160 × 160 y sus intensidades se dividen por 255."),
    (r"\bpooling\b|\bmaxpool|6\s*(x|por)\s*6|reduc.{0,12}3\s*(x|por)\s*3",
     "Max pooling recorre el mapa con ventanas de 2 × 2 y conserva el mayor valor de cada ventana. Con paso 2, un mapa de 6 × 6 se convierte en otro de 3 × 3, con una salida por bloque. Es un ejemplo de la operación aplicada a los mapas de características."),
    (r"\brelu\b|funcion.{0,12}activacion|\bmish\b|leaky",
     "Las convoluciones de nuestras CNN usan ReLU. Si la respuesta de un filtro es negativa, ReLU devuelve 0; si es positiva, conserva su valor. En nuestros ensayos LeakyReLU y Mish no redujeron las falsas alarmas del control conservando todos los aciertos. La neurona de salida usa sigmoide."),
    (r"sigmoid|probabil|confianza|certeza|que (es|significa) (un |el )?(puntaje|score)",
     "La sigmoide transforma la salida de cada red en un puntaje entre 0 y 1 para PNEUMONIA. Cada puntaje se compara con el umbral de esa red. Por ejemplo, un 0,90 es un puntaje alto del modelo; interpretarlo como un 90 % de certeza clínica requeriría demostrar su calibración."),
    (r"pesos? de (las? )?clase|peso.{0,12}(imagen|normal)|pondera|desbalance|desequilibr|50\s*50|balancead",
     "La CNN profunda equilibró sus lotes de entrenamiento usando 32 imágenes NORMAL y 32 PNEUMONIA en cada lote. La CNN de cuatro vistas ponderó los ejemplos por clase y proporción de la imagen. El test conservó su composición original de 234 normales y 390 neumonías."),
    (r"sesgo|\bbias\b",
     "El sesgo es un número que se suma a la combinación de entradas y pesos antes de la activación. Desplaza el punto en el que responde una neurona. Durante el entrenamiento el optimizador ajusta automáticamente tanto los pesos como los sesgos para reducir la pérdida."),
    (r"optimiza|gradiente|\badam\b|\bsgd\b|retropropaga|backprop|tasa de aprendizaje",
     "La retropropagación calcula cómo influye cada parámetro en la pérdida. El optimizador usa esos gradientes para actualizar pesos y sesgos en cada lote. La CNN original usó SGD. Las CNN de cuatro vistas y profunda usaron Adam con tasa decreciente. En la profunda también se limitó la norma global del gradiente a 1."),
    (r"\bpesos?\b|parametros entrenables",
     "Los pesos son los números que determinan cuánto influye cada entrada en la respuesta de una neurona o filtro. Se inicializan al crear la red y el optimizador los ajusta automáticamente con los lotes de entrenamiento. En la app se cargan los pesos aprendidos y se mantienen fijos."),
    (r"perdida|entropia|penaliza|binary crossentropy",
     "La pérdida compara el puntaje con la etiqueta real. En la entropía cruzada binaria, si la etiqueta es PNEUMONIA y la red da 0,10, la pérdida es −ln(0,10) ≈ 2,30. Para una etiqueta NORMAL se usa −ln(1 − puntaje). Un error con mucha confianza recibe una penalización mayor."),
    (r"\b(lotes?|batch|batches)\b",
     "Un lote es el grupo de imágenes con el que se calcula una actualización de los parámetros. La CNN profunda entrenó con lotes de 64, con 32 imágenes por clase. La CNN de cuatro vistas entrenó con lotes de 32."),
    (r"epocas?|checkpoint|cuando par|early stopping|sobreajuste|sobreajust",
     "Una época recorre los lotes programados para una ronda de entrenamiento. Guardar los pesos según el desempeño en validación ayuda a detectar cuándo seguir entrenando deja de ayudar. Para la app se guardaron la época 10 de la CNN original, la 12 de cuatro vistas y la 9 de la profunda."),
    (r"aplan|flatten|arquitectura|convoluc|filtros?|capas?|primera (cnn|red)|tercera (cnn|red)|cnn profunda",
     "La CNN original aplica tres convoluciones de 3 × 3 con 8, 16 y 32 filtros. Cada bloque usa ReLU y max pooling. Flatten reorganiza el último mapa en un vector y la salida sigmoide produce un puntaje. La CNN profunda añade dos convoluciones y la CNN de cuatro vistas analiza la placa con distintos encuadres."),
    (r"preentren|desde cero|transferencia|imagenet|tensorflow|keras",
     "Las tres redes aprendieron sus pesos durante el entrenamiento con el dataset del profesor. TensorFlow y Keras se usaron para definir las capas y calcular los gradientes. No se cargaron pesos de ImageNet ni de otro clasificador médico."),
    (r"etiqueta|\bclases\b|\bbinari[oa]\b|\b0 y 1\b|\buno y cero\b|que significa (normal|pneumonia)",
     "El objetivo es binario. La etiqueta 0 corresponde a NORMAL y la 1 a PNEUMONIA (neumonía). Son categorías del dataset para toda la imagen. Los valores de gris entre 0 y 1 son las intensidades de sus píxeles."),
)


def _explicar_asistente():
    from transformer_fichas import estado_modelo

    if estado_modelo()["disponible"]:
        motor = ("Para las preguntas educativas usamos un Transformer entrenado desde cero aquí. "
                 "Aprendió a relacionar las consultas con las fichas de la biblioteca. Ordena las fichas "
                 "y ayuda a elegir entre temas que tienen apoyo en las palabras de la pregunta. ")
    else:
        motor = ("Ahora está activa la búsqueda por términos. El Transformer requiere los pesos "
                 "correspondientes a la versión de las fichas. ")
    return (motor + "La respuesta muestra el texto de la ficha elegida y su fuente. "
            "Las consultas sobre una radiografía leen la predicción de las tres CNN para el archivo "
            "cargado. Las cifras del proyecto proceden de la evaluación guardada. Todo se ejecuta localmente.")


def _responder(q, metricas, regla, entrenamiento, resultado):
    if re.search(r"\b(transformers?|rag)\b", q):
        return _explicar_asistente()
    if re.search(r"\brx\s*0*3\b", q):
        return ("La RX-003 del ejemplo de test tiene etiqueta NORMAL, pero las tres CNN superaron sus umbrales. "
                "Sus puntajes fueron aproximadamente 0,876 (original), 0,987 (cuatro vistas) y 0,950 (profunda). "
                "Es un falso positivo verificado de esa imagen histórica.")
    if re.search(r"diagnost|lesion|zona|region del pulmon|localiza|limitaci|uso clinico|reemplaza|"
                 r"tratamiento|medicamento|tengo neumonia|soy normal|que parte|en que pulmon|"
                 r"puedo confiar|garanti|es confiable|medico|sirve para pacientes", q):
        return ("Las etiquetas corresponden a la radiografía pediátrica completa. Una salida PNEUMONIA indica "
                "que los tres puntajes superaron o igualaron sus cortes. El prototipo puede equivocarse, carece "
                "de localización anatómica validada y no determina el diagnóstico ni el tratamiento de una persona.")
    if re.search(r"antes|anterior|diferencia|mejor[oas]?|ganamos|perdimos|que cambio", q) and re.search(
            r"voto|modelo|cnn|alarma|84|86", q):
        return ("En el test histórico, la regla de dos redes tuvo 86 falsas alarmas. Al ajustar sus cortes quedaron 85 "
                "y al pedir también el tercer voto quedaron 84. Se conservaron las mismas 372 neumonías detectadas "
                "y ninguna normal acertada pasó a falsa alarma. En el control de desarrollo, ambas reglas tuvieron tres falsas alarmas.")
    # Una consulta explícita sobre el archivo toma el resultado de la sesión actual.
    if re.search(r"predij|prediccion (de|para|actual)|que (dio|salio)|resultado (de (mi|esta|la) |actual)|"
                 r"(puntaj|vot|clasific|predic|result).{0,35}(esta |mi |carg|subi|la imagen|la radiografia)|"
                 r"(esta |mi |cargad|subid).{0,28}(vot|clasific|puntaj|resultado)|"
                 r"que significa (el|este|mi) resultado", q):
        return resultado or "Carga una radiografía o activa el ejemplo del dataset para que pueda mostrar la clase y los tres votos de esa imagen."
    estadisticas = _metricas(q, metricas)
    if estadisticas:
        return estadisticas
    if re.search(r"\b(agente|asistente|fuentes|biblioteca)\b", q):
        return _explicar_asistente()
    if re.search(r"umbral|corte", q):
        if re.search(r"elig|fij|escog|ajust|seleccion|por que", q):
            return (regla + " Se eligió para cada red el corte más alto que mantenía las 675 detecciones previas "
                    "de neumonía en los datos de validación. El control formó parte de este ajuste.")
        return regla
    for patron, explicacion in LECCIONES:
        if re.search(patron, q):
            return explicacion
    if re.search(r"dataset|conjunto|entren|\bdatos\b|cuantas (imagenes|radiografias)|kaggle|"
                 r"de donde (salieron|provienen|proceden)", q):
        return (entrenamiento + " El conjunto original Chest X-Ray Images (Pneumonia) contiene 5.856 radiografías: "
                "5.216 en train, 16 en la validación original y 624 en test. "
                "[Dataset del profesor](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)")
    if re.search(r"test|prueba historica|para que sirve la validaci|diferencia.{0,22}validaci", q):
        return ("Los datos de entrenamiento ajustan pesos y sesgos. Los de validación ayudaron a escoger los pesos "
                "guardados y los umbrales. El test histórico tiene 624 imágenes, 234 normales y 390 neumonías. "
                "Sus métricas describen ese conjunto pediátrico.")
    if re.search(r"\bvotos?\b|unanim|mayoria|combin|consenso|tres (redes|cnn)|3 (redes|cnn)|"
                 r"como (funciona|clasifica|decide) (el |nuestro )?(modelo|sistema)|una (sola )?salida", q):
        return regla
    if re.fullmatch(r"(imagen|radiografia|prediccion|resultado|puntajes?)", q):
        return resultado or "Carga una radiografía o selecciona el ejemplo antes de consultar su predicción."
    return ("No tengo una respuesta documentada para esa pregunta sobre el proyecto. Prueba con los tres votos, "
            "la preparación de las imágenes, el entrenamiento o una métrica concreta. Para preguntas de salud, "
            "usa «Sobre la neumonía».")


def consultar_proyecto(pregunta, metricas, regla, entrenamiento, resultado, *, pregunta_anterior=""):
    """Responde a cada pregunta explícita con datos del sistema de tres CNN."""
    partes = [_normalizar(p) for p in re.split(r"[¿?\n]+", pregunta) if p.strip()]
    if len(partes) == 1 and re.fullmatch(
            r"(y )?((para|en|con|de) )?(las? |la clase )?(normal(?:es)?|neumonias?|pneumonia)", partes[0]):
        anterior = _normalizar(pregunta_anterior)
        tema = re.search(r"f\s?1(?: macro)?|precision|recall|sensibilidad|especificidad|exactitud", anterior)
        if tema:
            partes[0] = f"{tema.group()} {partes[0]}"
    respuestas = [_responder(q, metricas, regla, entrenamiento, resultado) for q in partes if q]
    return "\n\n".join(dict.fromkeys(respuestas))
