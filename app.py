import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# =======================
# Dataset ampliado
# =======================
data = {
    "titular": [
        # Solar
        "Instalan 2.000 paneles solares en escuelas rurales de Boyacá",
        "Nueva planta solar fotovoltaica abastecerá a 10.000 hogares",
        "Colombia supera récord de generación solar en 2025",
        "Proyecto solar en el desierto de La Tatacoa entra en operación",
        "Paneles solares flotantes serán instalados en embalses de Antioquia",

        # Eólica
        "Parque eólico marino genera electricidad para 50.000 familias",
        "Nuevas turbinas eólicas en La Guajira duplican capacidad energética",
        "Avanza la construcción del mayor parque eólico terrestre del país",
        "Vientos del Caribe impulsan nueva planta eólica",
        "Energía eólica abastecerá fábricas en el Atlántico",

        # Biomasa
        "Planta de biomasa transforma residuos de palma en energía",
        "Aprovechan cáscaras de arroz para generar electricidad en el Meta",
        "Nueva central de biomasa abastecerá industrias locales",
        "Proyecto de biomasa con caña de azúcar reduce emisiones",
        "Producción de biogás a partir de residuos agrícolas",

        # Hidroeléctrica
        "Ampliación de central hidroeléctrica en Antioquia entra en fase final",
        "Nueva represa hidroeléctrica aportará energía a tres departamentos",
        "Proyecto hidroeléctrico reduce dependencia de combustibles fósiles",
        "Hidroeléctrica en el Cauca genera energía limpia para comunidades",
        "Pequeña hidroeléctrica abastecerá veredas en Santander",

        # Geotérmica
        "Exploración geotérmica en Nariño avanza con éxito",
        "Hospital en Pasto será abastecido con energía geotérmica",
        "Proyecto geotérmico aprovecha calor volcánico para producir electricidad",
        "Colombia estudia potencial geotérmico en zonas volcánicas",
        "Inician perforaciones para energía geotérmica en el Tolima",
    ],
    "categoria": (
        ["solar"]*5 +
        ["eólica"]*5 +
        ["biomasa"]*5 +
        ["hidroeléctrica"]*5 +
        ["geotérmica"]*5
    )
}

df = pd.DataFrame(data)

# =======================
# Entrenamiento del modelo
# =======================
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(df["titular"])
model = MultinomialNB()
model.fit(X_tfidf, df["categoria"])

# Info adicional de cada tipo de energía
info_energia = {
    "solar": {
        "nombre": "Energía Solar ☀",
        "descripcion": "Aprovecha la radiación del sol mediante paneles fotovoltaicos o sistemas térmicos para generar electricidad o calor.",
        "ejemplo": "Planta solar de Cauchari en Argentina, con más de 1 millón de paneles.",
        "color": "#FFD700"
    },
    "eólica": {
        "nombre": "Energía Eólica 🌬",
        "descripcion": "Utiliza la fuerza del viento para mover turbinas que generan electricidad.",
        "ejemplo": "Parque Eólico Gecama en España, con 312 MW de capacidad.",
        "color": "#ADD8E6"
    },
    "biomasa": {
        "nombre": "Energía de Biomasa 🌱",
        "descripcion": "Convierte materia orgánica como residuos agrícolas o forestales en electricidad, calor o biocombustibles.",
        "ejemplo": "Central de biomasa de Soria en España, que usa astillas de madera.",
        "color": "#90EE90"
    },
    "hidroeléctrica": {
        "nombre": "Energía Hidroeléctrica 💧",
        "descripcion": "Genera electricidad aprovechando la fuerza del agua en movimiento, usualmente mediante presas.",
        "ejemplo": "Represa de Itaipú entre Brasil y Paraguay, una de las mayores del mundo.",
        "color": "#87CEFA"
    },
    "geotérmica": {
        "nombre": "Energía Geotérmica 🌋",
        "descripcion": "Utiliza el calor interno de la Tierra para generar electricidad o calefacción.",
        "ejemplo": "Planta geotérmica de Nesjavellir en Islandia.",
        "color": "#FFA07A"
    }
}

# =======================
# Interfaz en Streamlit
# =======================
st.set_page_config(page_title="Clasificador de Noticias", page_icon="📰", layout="wide")
st.title("📰 Clasificador de Noticias sobre Energías Renovables")
st.write("Esta aplicación clasifica titulares de noticias y te muestra información sobre el tipo de energía renovable.")

# Entrada individual
st.header("🔍 Clasificación de un titular")
titular = st.text_input("Escribe el titular de la noticia:")

if st.button("Clasificar"):
    if titular.strip():
        X_new = vectorizer.transform([titular])
        pred = model.predict(X_new)[0]
        prob = model.predict_proba(X_new)[0]
        info = info_energia[pred]

        st.markdown(f"### {info['nombre']}")
        st.write(info['descripcion'])
        st.info(f"Ejemplo real: {info['ejemplo']}")
        st.markdown(f"**Probabilidad por categoría:**")
        for cat, p in zip(model.classes_, prob):
            st.write(f"- {cat}: {p*100:.2f}%")
    else:
        st.warning("Por favor, escribe un titular.")

# Clasificación en lote
st.header("📂 Clasificar titulares desde un archivo CSV")
archivo = st.file_uploader("Sube un archivo CSV con una columna llamada 'titular'", type=["csv"])
if archivo is not None:
    df_subida = pd.read_csv(archivo)
    if "titular" in df_subida.columns:
        X_lote = vectorizer.transform(df_subida["titular"])
        df_subida["categoria_predicha"] = model.predict(X_lote)
        st.write(df_subida)
        st.download_button("Descargar resultados", df_subida.to_csv(index=False), file_name="resultados.csv")
    else:
        st.error("El archivo debe tener una columna llamada 'titular'.")
