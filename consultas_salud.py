"""Búsqueda local en la biblioteca de neumonía, con citas de las fichas revisadas."""
import json
import math
import re
import unicodedata
from collections import Counter
from difflib import get_close_matches
from functools import lru_cache
from pathlib import Path

from transformer_fichas import ranquear


ARCHIVO = Path(__file__).resolve().parent / "recursos" / "fuentes_neumonia.json"


def normalizar(texto):
    letras = unicodedata.normalize("NFKD", texto.casefold())
    return " ".join(re.findall(r"[a-z0-9]+", "".join(c for c in letras if not unicodedata.combining(c))))


# Los términos de cortesía y los verbos de apoyo no identifican el tema.
COMUNES = set("""
    a al algun alguna algunos algunas ante aqui asi acerca aparte bajo buen buenas cada con contra cual
    cuales cualquier cuando cuyo da dan dar de del desde debo debe deben deberia decir dime donde
    durante e el ella en entre era es esa ese eso esta estan este esto estos existe existen fue guia
    ha haber hace hacen hacer hacia han hasta hay hola informacion la las le les lo los manera como
    mas me mi mis mismo mucha muy necesito nos o otro otros otra otras para parte pasa pasar pero por
    porque podria podemos poder posible principales puede pueden puedo que quien quienes quiero respecto
    realmente relacionada relacionadas saber se segun ser si sin sobre son su sus suele suelen tambien
    tanto te tiene tienen tengo toda todas todo todos tras tu un una unas uno unos usted ver y ya siempre
    enfermedad enfermedades favor quisiera gustaria hablame cuentan cuenta explican explicas explica normalmente
    podrias podrian explicar explicarme explicame describir describirme interesa interesan interesaria
    podra podran exactamente principales contame generalmente presenta presentan presentarse aparecen
    presentar aparezca recomienda recomiendan deberiamos llamado llamada dicen elabora saber
    puedes dime seria sirve sirven importa importan sucede ocurrir pasa pasar suele suelen comun comunes
    habituales frecuente frecuentes persona personas alguien paciente pacientes cositas acerca profesor
    verdad cuento cuentas sabes podre resulta hacen quiere decir respecto relacion existen algunos determinada
    consiste consisten significa muestren muestra mostrar poblacion suele observarse observan se observan
    palabra palabras sencillo sencillos sencilla sencillas facilmente terminos generales cotidiano coloquial
    dice dicen profe hablando hablamos general explicarle contarle contaria ven observa
""".split())

# Una misma idea se expresa con sustantivos, conjugaciones y vocabulario cotidiano.
# Los demás términos se aprenden del propio índice local de fichas.
VARIANTES = {
    "neumonia": "neumonia neumonias pulmonia pulmonias neumomia neumoniaa pneumonia neomonia",
    "sintoma": "sintoma sintomas signo signos manifestacion manifestaciones manifiesta manifiestan manifestarse molestias malestares malestar sintomatologia senal senales",
    "infancia": "nino ninos nina ninas pediatrico pediatrica pediatricos pediatricas infantil infantiles infancia pequenos pequenas peques chiquitos chiquitas",
    "bebe": "bebe bebes lactante lactantes neonatos neonato reciennacido guagua guaguas bebito bebita bebecito bebecita",
    "adulto": "adulto adulta adultos adultas",
    "mayor": "mayor mayores anciano anciana ancianos ancianas envejecimiento viejito viejita viejitos abuelo abuela abuelos abuelas vejez",
    "prevencion": "prevencion prevenir previene previenen prevengo prevenirla prevengamos preventiva preventivo preventivas preventivos evitar evitarla evitamos evito cuido cuidarse proteger protegerse protegerlos protegen",
    "tratamiento": "tratar trata tratan tratarla tratado tratamiento tratamientos terapia terapias curar curan cura curarla combate combatir",
    "medicamento": "medicamento medicamentos medicacion medicina medicinas farmaco farmacos",
    "antibiotico": "antibiotico antibioticos antibioterapia",
    "causa": "causa causas causan causante causantes causada causado causadas causados causar provoca provocan provocar origen origina originar originan origino ocasiona ocasionan ocasionar desencadena desencadenan desencadenar produce producen razon razones motivo motivos etiologia",
    "contagio": "contagio contagios contagia contagian contagiar contagiosa contagioso contagiarse pega pegue pegar pegarse transmite transmiten transmitir transmision transmitirse propaga propagacion propagar propagarla",
    "germen": "germen germenes microorganismo microorganismos microbio microbios patogeno patogenos",
    "riesgo": "riesgo riesgos vulnerable vulnerables propenso propensos propensa propensas susceptibilidad susceptible susceptibles",
    "diagnostico": "diagnostico diagnostica diagnostican diagnosticar diagnosticarla detecta detectan deteccion confirmar confirma comprueba comprobar detectarla detectar diagnosticada",
    "examen": "examen examenes estudio estudios estudiar prueba pruebas test tests analisis analizar analizan analizado",
    "radiografia": "radiografia radiografias radiografico radiologica placa placas rx",
    "virus": "virus viral virica virico viricos viricas virales",
    "bacteria": "bacteria bacterias bacteriano bacteriana bacterianas bacterianos",
    "hongo": "hongo hongos fungica fungicas fungico micotica",
    "vacuna": "vacuna vacunas vacunacion vacunarse vacunado vacunada inmunizar inmunizacion inmunizado inmunizada inmunizadas inmunizados",
    "recuperacion": "recuperacion recuperarse recupera recuperarme recuperarnos convalecencia recuperando recuperan restablecerse reponerse",
    "duracion": "duracion tarda tardo tardar demora demorarse durar dura dias semanas meses semana mes plazo plazos",
    "complicacion": "complicacion complicaciones complicarse complica consecuencias secuelas",
    "alarma": "alarma alarmas emergencia emergencias urgencia urgencias urgente alarmante peligro inmediata inmediato",
    "hospital": "hospital hospitales hospitalizacion hospitalizada hospitalizar ingreso ingresar internacion internamiento internan hospitalizado internado",
    "diferencia": "diferencia diferencias diferenciar diferencio distingue distinguir distincion comparar comparacion versus vs",
    "aspiracion": "aspiracion aspirar aspirado atragantamiento broncoaspiracion",
    "adquirir": "adquirir adquirida adquirio adquiere adquirieron contraida contrajo",
    "oximetria": "oximetria oximetro pulsioximetro pulsioximetria",
    "tomografia": "tomografia tomografias tac tc escaner",
    "rapido": "rapido rapidos rapida rapidas acelerada acelerado",
    "fumar": "fumar fuma fumadores fumador tabaco tabaquismo cigarrillo cigarrillos",
    "faltaire": "disnea",
    "respirar": "respirar respira respiran respiracion respirando respiratoria respiratorio inspirar inspiracion",
    "confusion": "confuso confusa confundido confundida confusion desorientacion desorientado desorientada desorientarse desorienta",
    "prematuro": "prematuridad prematuro prematuros prematura pretermino prematuramente",
    "esputo": "esputo expectoracion expectora expectorar flema flemas",
}
ALIAS = {variante: termino for termino, variantes in VARIANTES.items() for variante in variantes.split()}
EDAD = {"infancia", "bebe", "adulto", "mayor"}
GENERICOS = {"neumonia", "tipo", "tipos", "diferencia", "tiempo", "frecuencia", "general",
             "caso", "casos", "hacer", "significa", "pulmon", "pulmone", "problema", "enferma", "afecta"}
FORMAS_CORTAS = {"que es", "en que consiste", "que significa", "que quiere decir",
                 "definicion", "explicame", "hablame", "neumonia", "pulmonia", "pneumonia"}


def terminos(texto):
    palabras = set()
    limpio = re.sub(r"\brecien nacid[oa]s?\b", "bebe", normalizar(texto))
    if re.search(r"\buna persona a otra|de persona a persona|entre personas\b", limpio):
        palabras.add("contagio")
    if re.search(r"\b(?:bajar|disminuir|reducir|reducen|minimizar) (?:la probabilidad|el riesgo)|"
                 r"\b(?:lavado|lavar\w*) (?:de |las )?manos\b", limpio):
        palabras.add("prevencion")
    if re.search(r"\b(?:tom[ae]r?|cog[ea]r?) (?:el )?aire\b", limpio):
        palabras.add("respirar")
    if re.search(r"\b(?:nac\w*|parto)\b.{0,28}\bantes de (?:tiempo|termino|las 37 semanas)\b", limpio):
        palabras.add("prematuro")
    if re.search(r"\b(?:expuls\w*|arroja\w*|sac\w*)\b.{0,25}\b(?:al|cuando) tos\w*\b", limpio):
        palabras.add("esputo")
    if re.search(r"\b(?:recuper\w*|volver a|recobr\w*)\b.{0,24}\b(?:energia|fuerzas?|rutina|actividades)\b", limpio):
        palabras.add("recuperacion")
    if ("detalle" in limpio or "detallad" in limpio) and re.search(r"\b(placa|rayos x|radiografia)\b", limpio):
        palabras.add("tomografia")
    for palabra in limpio.split():
        if palabra in COMUNES or len(palabra) < 2:
            continue
        if palabra in ALIAS:
            palabras.add(ALIAS[palabra])
        elif palabra.startswith(("neum", "pulm", "pneu")) and get_close_matches(palabra, ("neumonia", "pulmonia", "pneumonia"), cutoff=.82):
            palabras.add("neumonia")
        elif len(palabra) > 4 and palabra.endswith("s") and not palabra.endswith(("sis", "tis")):
            palabras.add(palabra[:-1])
        else:
            palabras.add(palabra)
    return palabras


@lru_cache(maxsize=2)
def _cargar(ruta, modificacion, tamano):
    """El sello invalida el índice si el docente agrega o edita fichas."""
    datos = json.loads(ruta.read_text(encoding="utf-8-sig"))
    fichas, frecuencia, vocab_textos = [], Counter(), set()
    for original in datos["fichas"]:
        ficha = dict(original, fuente=datos["fuentes"][original["fuente"]])
        ejemplos = [terminos(t) - {"neumonia"} for t in (ficha["titulo"], *ficha["consultas"])]
        vocabulario = set().union(*ejemplos)
        texto = terminos(ficha["texto"]) - {"neumonia"}
        frecuencia.update(vocabulario)
        vocab_textos.update(texto)
        fichas.append((ficha, ejemplos, vocabulario))
    pesos = {p: math.log(1 + (len(fichas) + .5) / (n + .5)) for p, n in frecuencia.items()}
    return fichas, pesos, vocab_textos


def _indice():
    stat = ARCHIVO.stat()
    return _cargar(ARCHIVO, stat.st_mtime_ns, stat.st_size)


def biblioteca():
    fichas, pesos, _ = _indice()
    return fichas, pesos


def es_consulta_personal(pregunta):
    q = normalizar(pregunta)
    prescripcion = (
        r"\b(dosis|dosificacion|miligramos?|mililitros?|mg|ml|receta|recetame|automedicar\w*|diagnosticame)\b|"
        r"\b(cuant[oa]s?|que cantidad de)\b.{0,15}\b(gotas?|cucharaditas?|cucharadas?|pastillas?|tabletas?|capsulas?)\b|"
        r"\b(cuantos dias|cuantas veces|cada cuant[oa]s?)\b.{0,50}\b(antibiotico\w*|medicamento\w*|pastilla\w*|amoxicilina)\b|"
        r"\b(que|cual|cuanto|debo|deberia|puedo|puede|recomiendas?)\b.{0,36}\b(tomarme|tomar|darle|le doy|le puedo dar|me tomo|debo tomar|debo darle|debo usar|debo usarlo)\b.{0,45}"
        r"\b(antibiotico\w*|medicamento\w*|pastilla\w*|amoxicilina|azitromicina|ibuprofeno|paracetamol|medicina|jarabe|dosis)\b|"
        r"\b(antibiotico\w*|medicamento\w*|amoxicilina|pastilla\w*|jarabe)\b.{0,45}\b(le doy|debo tomar|puedo tomar|tomarme|darle|me tomo|me recomiend\w*)\b"
        r"|\b(antibiotico\w*|medicamento\w*|amoxicilina|jarabe\w*|pastilla\w*)\b.{0,32}\b(deberia|tengo que|puedo|debo) (tomar|usar|dar)\b"
        r"|\b(que (le doy|le puedo dar|debo tomar|tomo)|que puedo darle)\b"
    )
    diagnostico_individual = (
        r"\b(tengo|tenemos|tendre|tendra|sera|podria (ser|tener)|tiene mi (bebe|hij[oa])|mi (bebe|hij[oa]) tiene) (una )?(neumonia|pulmonia)\b|"
        r"\b(mi|esta|esa) (placa|radiografia|imagen|rx)\b|\bmis sintomas\b|"
        r"\b(me (?:confirm\w*|diagnostic\w*)|(?:si|que) lo mio es|(?:mi|mis) (?:tos|fiebre|dolores))\b|"
        r"\b(tengo|mi (bebe|hij[oa]|madre|padre|abuela|abuelo) (tiene|esta con))\b.{0,30}\b(fiebre|tos|dolor|vomitos|sintomas)\b"
    )
    return bool(re.search(prescripcion, q) or re.search(diagnostico_individual, q))


def _peligro_expreso(q):
    return bool(re.search(
        r"\b((?:no|apenas|casi no) (pued[eo]|logr[oa]|consig[ou]e?) respirar|no respira|"
        r"labios (azules|morados|azulados)|"
        r"se (esta )?ahoga(ndo)?|dificultad (grave|severa|intensa) (para|al) respirar|"
        r"(muchisimo|mucho) (trabajo|esfuerzo) (para|al) respirar|se le hunden las costillas)\b", q))


def _fuera_de_alcance(q, vocabulario):
    # Ninguna ficha contiene estadísticas locales, cotizaciones ni esquemas de un país.
    sin_cobertura = re.search(
        r"\b(precio|precios|cuesta|cuestan|cuanto vale|cotiza\w*|costo|costos|politica|presidente|capital|futbol|bitcoin|programacion|clima de hoy)\b|"
        r"\b(cuant[oa]s? (casos|muertes|personas|ninos|hospitalizaciones|radiografias)|incidencia|prevalencia|tasa de mortalidad|estadisticas?|cifras?|porcentaje (de|en))\b|"
        r"\b(horarios?|direccion|telefono|a que hora (abre|cierra)\w*)\b.{0,60}\b(hospital|clinica|farmacia|urgencias)\b|"
        r"\bcuant[oa]s?\b.{0,45}\b(hubo|ocurrieron|reportaron|notificaron|registraron)\b|"
        r"\b(quien|cuando|en que ano)\b.{0,20}\b(invento|inventaron|descubrio|descubrieron|descubrimiento|patento|historia)\b|"
        r"\b(hospital|clinica|farmacia)s? (cercan[oa]s?|abiert[oa]s?|en (mi|bogota|colombia))\b|"
        r"\b(dengue|rabia|viruela|malaria|alzheimer|apendicitis|zika|ebola|fiebre amarilla|turismo|viaje internacional)\b", q)
    if sin_cobertura:
        return True
    if "neumonia" in terminos(q):
        return False
    # «Síntomas del [tema]», «cómo prevenir [tema]», etc. implican un objeto
    # explícito. Su nombre debe estar cubierto por la biblioteca.
    objeto = re.search(r"\b(?:sintomas?|tratamiento|prevencion|vacunas?|contagio|prevenir|"
                       r"previene|trata|cura|contagia)(?: (?:de|del|para))? (?:la |el |los |las )?"
                       r"([a-z]+)\b", q)
    if objeto:
        conceptos = terminos(objeto[1]) - EDAD
        return bool(conceptos - vocabulario)
    return False


def _con_contexto(palabras, previo, fichas):
    if not previo or not palabras or not palabras <= EDAD | GENERICOS:
        return palabras
    ficha = next((f for f, _, _ in fichas if f["id"] == previo), None)
    if not ficha:
        return palabras
    temas = terminos(ficha["id"].replace("_", " ")) - EDAD - GENERICOS
    return palabras | temas


def _solicita_datos_externos(q):
    """Las fichas describen conceptos. No contienen agendas, precios ni tasas locales."""
    servicio = re.search(r"\b(hospital\w*|clinic\w*|centro de salud|laboratorio\w*|farmacia\w*|consultorio\w*)\b", q)
    gestion = re.search(r"\b(citas?|turnos?|cupos?|disponib\w*|reserv\w*|agend\w*|abiert\w*|atiend\w*|"
                        r"horarios?|direcciones?|domicilios?|ubicad\w*|mas cercan\w*|telefonos?)\b", q)
    frecuencia = re.search(r"\b(que tan (?:comun|frecuente) es|con que frecuencia (?:ocurre|aparece|se da)|"
                           r"cuantos? (?:de cada|por cada)|que proporcion|probabilidad exacta)\b", q)
    return bool((servicio and gestion) or frecuencia)


def _candidatas_lexicas(fragmento, previo, fichas, pesos, vocab_textos):
    # Corregir solo variantes cercanas de términos del corpus. Las cantidades y los
    # nombres cortos permanecen intactos para evitar coincidencias accidentales.
    palabras = terminos(fragmento)
    menciona_enfermedad = "neumonia" in palabras
    palabras.discard("neumonia")
    if (menciona_enfermedad and palabras & EDAD
            and re.search(r"\b(afect\w*|manifest\w*|present\w*|mostr\w*)\b", fragmento)):
        palabras.add("sintoma")
    if not previo:
        literales = [f for f, _, _ in fichas
                     if any(fragmento == normalizar(ejemplo) for ejemplo in f["consultas"])]
        if len(literales) == 1 and (palabras - GENERICOS - EDAD or len(fragmento.split()) >= 3):
            return [(literales[0], 1.)]
    if menciona_enfermedad and palabras <= GENERICOS:
        return [(f, 1.) for f, _, _ in fichas if f["id"] == "definicion"]
    if re.search(r"\bpor que (da|se (produce|desarrolla)|aparece|ocurre)\b", fragmento):
        palabras.add("causa")
    if re.search(r"\b(que (es|son)|definicion|que significa|en que consiste)\b", fragmento):
        if menciona_enfermedad and (palabras <= EDAD | GENERICOS or not palabras):
            return [(f, 1.) for f, _, _ in fichas if f["id"] == "definicion"]
    if not palabras and (menciona_enfermedad or fragmento in FORMAS_CORTAS and previo == "definicion"):
        return [(f, 1.) for f, _, _ in fichas if f["id"] == "definicion"]
    palabras = _con_contexto(palabras, previo, fichas)
    conocidos = set()
    desconocidos = set()
    for p in palabras:
        if p in pesos or p in vocab_textos or p in EDAD:
            conocidos.add(p)
        elif len(p) >= 5:
            vecinos = get_close_matches(p, pesos.keys(), n=1, cutoff=.85)
            if vecinos:
                conocidos.add(vecinos[0])
            else:
                desconocidos.add(p)
        else:
            desconocidos.add(p)
    # Un concepto específico más una palabra de apoyo desconocida es frecuente en
    # el habla. Un adjetivo genérico o la edad por sí solos no acreditan un tema.
    nucleos = conocidos - EDAD - GENERICOS
    if not nucleos:
        return []
    if len(desconocidos) > 2 * len(nucleos) + int(menciona_enfermedad) + len(conocidos & EDAD):
        return []

    candidatas = []
    for ficha, ejemplos, vocabulario in fichas:
        comunes = conocidos & vocabulario
        if not comunes & nucleos:
            continue
        if ficha["id"] == "urgencias" and not ({"alarma", "ayuda"} & conocidos or _peligro_expreso(fragmento)
                or re.search(r"\b(cuando|debo) (ir|acudir|consultar)\b", fragmento)):
            continue
        mejor_ejemplo = 0.
        for ejemplo in ejemplos:
            solapamiento = ejemplo & conocidos
            if not solapamiento & nucleos:
                continue
            suma = sum(pesos.get(p, 1.) for p in solapamiento)
            cobertura = suma / sum(pesos.get(p, 1.) for p in ejemplo)
            mejor_ejemplo = max(mejor_ejemplo, suma * (.55 + .45 * cobertura))
        en_texto = conocidos & terminos(ficha["texto"])
        tema_ficha = terminos(ficha["id"].replace("_", " ")) - EDAD
        titulo_tema = tema_ficha & nucleos
        peso_tema = len(titulo_tema) / max(1, len(tema_ficha))
        valor = (mejor_ejemplo + .12 * sum(pesos.get(p, 1.) for p in comunes)
                 + .9 * peso_tema * sum(pesos.get(p, 1.) for p in titulo_tema)
                 + .06 * len(en_texto & nucleos))
        # Si una pregunta omite la edad, favorecer la ficha general frente a la
        # de un grupo concreto cuando existe evidencia para ambas.
        publico = terminos(ficha["id"].replace("_", " ")) & EDAD
        if publico and not (conocidos & EDAD):
            valor *= .72
        if valor > 0:
            candidatas.append((ficha, valor))
    return sorted(candidatas, key=lambda par: par[1], reverse=True)


def _seleccionar(fragmento, candidatas, usar_transformer):
    """El encoder solo puede desempatar fichas que la consulta sustenta."""
    primera, maximo = candidatas[0]
    orden = ranquear(fragmento, archivo_corpus=ARCHIVO) if usar_transformer else []
    if not orden:
        return primera, "lexico"
    favorita, puntaje = orden[0]
    if favorita == primera["id"]:
        return primera, "acuerdo"
    margen = puntaje - orden[1][1] if len(orden) > 1 else puntaje
    if puntaje >= .65 and margen >= .20:
        for ficha, valor in candidatas:
            if ficha["id"] == favorita and valor >= .80 * maximo:
                return ficha, "desempate"
    return primera, "lexico"


def consultar_con_ruta(pregunta, tema_anterior=None, *, usar_transformer=True, verificar_cobertura=True):
    """Devuelve fichas citadas y los métodos que intervinieron en esta respuesta."""
    fichas, pesos, vocab_textos = _indice()
    q = normalizar(pregunta[:1500])
    if not q:
        return [], []
    if _peligro_expreso(q):
        return [f for f, _, _ in fichas if f["id"] == "urgencias"], ["urgencia"]
    if es_consulta_personal(pregunta) or _fuera_de_alcance(q, vocab_textos | set(pesos)):
        return [], []
    cobertura = usar_transformer and verificar_cobertura
    if cobertura and _solicita_datos_externos(q):
        return [], []
    # Preguntas con varios propósitos, como «qué síntomas da y cómo se previene».
    # Las enumeraciones («virus y bacterias») se consultan juntas.
    separar_temas = (r"\b(?:y|ademas|tambien)\s+(?=(?:como|que|quien|cual|por que|cuando|en que|"
                     r"prevencion|tratamiento|sintomas|causas|factores|complicaciones|transmision)\b)")
    partes = [parte for oracion in re.split(r"[;?]+", pregunta[:1500])
              for parte in re.split(separar_temas, normalizar(oracion))]
    salida, rutas = [], []
    for parte in partes:
        parte = normalizar(parte)
        if not parte:
            continue
        candidatas = _candidatas_lexicas(parte, tema_anterior, fichas, pesos, vocab_textos)
        if not candidatas:
            continue
        ficha, ruta = _seleccionar(parte, candidatas, usar_transformer)
        if cobertura and not tema_anterior and ruta == "lexico":
            # Un término breve como «ve» puede coincidir por azar. Conservar las
            # abreviaturas médicas y las coincidencias aproximadas del recuperador.
            conceptos = terminos(parte)
            vocab_ficha = next(vocab for original, _, vocab in fichas if original["id"] == ficha["id"])
            claros = {p for p in conceptos if (len(p) > 2 or p in {"rx", "tc"})
                      and (p in vocab_ficha or p == "neumonia")}
            aproximados = any(len(p) >= 5 and get_close_matches(p, vocab_ficha, n=1, cutoff=.85)
                              for p in conceptos)
            if not claros and not aproximados:
                continue
        if ficha not in salida:
            salida.append(ficha)
            rutas.append(ruta)
            if len(salida) == 3:
                break
    # Para una comparación de manifestaciones por edad, mostrar ambos contextos.
    edades = terminos(q) & EDAD
    if (salida and salida[0]["id"] in {"sintomas_ninos", "sintomas"}
            and edades & {"bebe", "infancia"} and "adulto" in edades):
        por_id = {f["id"]: f for f, _, _ in fichas}
        return [por_id["sintomas_ninos"], por_id["sintomas"],
                *(f for f in salida if f["id"] not in {"sintomas_ninos", "sintomas"})], [*rutas, "edades"]
    return salida, rutas


def consultar(pregunta, tema_anterior=None):
    """Recupera fichas citadas. El contexto opcional es el ID de la ficha anterior."""
    return consultar_con_ruta(pregunta, tema_anterior)[0]
