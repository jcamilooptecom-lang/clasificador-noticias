import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

# ================================
# 1. Dataset interno para entrenamiento
# ================================
data = {
    "titular": [
        # Solar
        "Nuevo parque solar en La Guajira",
        "Instalan paneles solares en colegios rurales",
        "Planta solar abastecerá 10.000 hogares",
        "Colombia supera récord de energía solar",
        # Eólica
        "Avances en energía eólica en el Caribe colombiano",
        "Instalan turbinas eólicas en La Guajira",
        "Vientos del Caribe impulsan parque eólico",
        "Parque eólico marino genera electricidad",
        # Biomasa
        "Producción de biogás a partir de residuos agrícolas",
        "Planta de biomasa transforma desechos en energía",
        "Aprovechan cáscaras de arroz para generar electricidad",
        "Proyecto de biomasa reduce emisiones de CO2",
        # Hidroeléctrica
        "Ampliación de central hidroeléctrica en Antioquia",
        "Nueva represa hidroeléctrica abastece tres departamentos",
        "Proyecto hidroeléctrico reduce uso de combustibles fósiles",
        "Hidroeléctrica en el Cauca genera energía limpia",
        # Geotérmica
        "Perforaciones para energía geotérmica en Nariño",
        "Hospital en Pasto será abastecido con energía geotérmica",
        "Proyecto geotérmico aprovecha calor volcánico",
        "Colombia estudia potencial geotérmico en zonas volcánicas"
    ],
    "categoria": [
        "solar", "solar", "solar", "solar",
        "eólica", "eólica", "eólica", "eólica",
        "biomasa", "biomasa", "biomasa", "biomasa",
        "hidroeléctrica", "hidroeléctrica", "hidroeléctrica", "hidroeléctrica",
        "geotérmica", "geotérmica", "geotérmica", "geotérmica"
    ]
}

df = pd.DataFrame(data)

# ================================
# 2. Entrenar el modelo
# ================================
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])
model.fit(df['titular'], df['categoria'])

# Guardar modelo entrenado
joblib.dump(model, 'modelo_clasificador.pkl')

# ================================
# 3. Información de cada tipo de energía
# ================================
info_energias = {
    "solar": {
        "definicion": "La energía solar aprovecha la radiación del Sol para generar electricidad o calor.",
        "dato": "En solo una hora, el Sol produce suficiente energía para abastecer al mundo entero durante un año.",
        "imagen": "https://upload.wikimedia.org/wikipedia/commons/0/0c/Solar_panels_in_Ohio.jpg"
    },
    "eólica": {
        "definicion": "La energía eólica convierte la fuerza del viento en electricidad mediante aerogeneradores.",
        "dato": "Un solo aerogenerador puede alimentar hasta 1.500 hogares al año.",
        "imagen": "https://upload.wikimedia.org/wikipedia/commons/3/3a/Wind_Turbine_Farm.jpg"
    },
    "biomasa": {
        "definicion": "La biomasa utiliza materia orgánica como residuos agrícolas o forestales para producir energía.",
        "dato": "La biomasa es la fuente renovable más antigua usada por el ser humano.",
        "imagen": "https://upload.wikimedia.org/wikipedia/commons/3/3f/Biomass_powerplant.jpg"
    },
    "hidroeléctrica": {
        "definicion": "La energía hidroeléctrica genera electricidad aprovechando la fuerza del agua en movimiento.",
        "dato": "Es la fuente renovable más utilizada en el mundo, representando más del 16% de la electricidad global.",
        "imagen": "https://upload.wikimedia.org/wikipedia/commons/6/6b/ThreeGorgesDam-China2009.jpg"
    },
    "geotérmica": {
        "definicion": "La energía geotérmica aprovecha el calor interno de la Tierra para producir electricidad o calefacción.",
        "dato": "Islandia genera casi el 100% de su electricidad usando fuentes renovables, gran parte geotérmica.",
        "imagen": "https://upload.wikimedia.org/wikipedia/commons/d/d8/Nesjavellir_Geothermal_Power_Plant.jpg"
    }
}

# ================================
# 4. Interfaz de la aplicación
# ================================
st.set_page_config(page_title="Clasificador de Energías Renovables", page_icon="⚡", layout="wide")

st.title("⚡ Clasificador de Noticias de Energías Renovables")
st.write("Escribe un titular y descubre de qué tipo de energía renovable se trata, junto con datos curiosos e información útil.")

# Entrada del usuario
titular_usuario = st.text_input("✏ Ingresa el titular de la noticia:")

if titular_usuario:
    # Cargar modelo
    modelo_cargado = joblib.load('modelo_clasificador.pkl')
    categoria_predicha = modelo_cargado.predict([titular_usuario])[0]

    st.success(f"**Categoría detectada:** {categoria_predicha.capitalize()} ✅")

    # Mostrar información de la energía detectada
    st.subheader("ℹ Información sobre esta energía")
    st.write(f"**Definición:** {info_energias[categoria_predicha]['definicion']}")
    st.write(f"**Dato curioso:** {info_energias[categoria_predicha]['dato']}")
    st.image(info_energias[categoria_predicha]['imagen'], use_column_width=True)
