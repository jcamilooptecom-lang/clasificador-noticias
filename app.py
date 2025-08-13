import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# =======================
# Dataset de ejemplo
# =======================
data = {
    "titular": [
        "Nuevo parque solar en La Guajira",
        "Avances en energía eólica en el Caribe colombiano",
        "Producción de biogás a partir de residuos agrícolas",
        "Ampliación de central hidroeléctrica en Antioquia",
        "Perforaciones para energía geotérmica en Nariño",
        "Colombia inaugura granja solar de 50 MW",
        "Instalación de turbinas eólicas en la costa atlántica",
        "Proyecto de biomasa aprovecha cáscaras de café",
        "Nueva represa hidroeléctrica reduce emisiones",
        "Calor volcánico se usará para generar electricidad",
        "Paneles solares abastecen comunidad rural",
        "Parque eólico marino inicia operaciones",
        "Biomasa de caña de azúcar genera electricidad",
        "Ampliación de planta hidroeléctrica en el Cauca",
        "Energía geotérmica abastecerá hospital en Pasto"
    ],
    "categoria": [
        "solar",
        "eólica",
        "biomasa",
        "hidroeléctrica",
        "geotérmica",
        "solar",
        "eólica",
        "biomasa",
        "hidroeléctrica",
        "geotérmica",
        "solar",
        "eólica",
        "biomasa",
        "hidroeléctrica",
        "geotérmica"
    ]
}

df = pd.DataFrame(data)

# =======================
# Entrenamiento del modelo
# =======================
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(df["titular"])
model = MultinomialNB()
model.fit(X_tfidf, df["categoria"])

# =======================
# Interfaz en Streamlit
# =======================
st.set_page_config(page_title="Clasificador de Noticias", page_icon="📰")
st.title("📰 Clasificador de Noticias sobre Energías Renovables")
st.write("Escribe un titular y el sistema lo clasificará en: solar, eólica, biomasa, hidroeléctrica o geotérmica.")

# Entrada de texto
titular = st.text_input("✏️ Titular de la noticia:")

if st.button("Clasificar"):
    if titular.strip():
        X_new = vectorizer.transform([titular])
        pred = model.predict(X_new)[0]
        st.success(f"La noticia es de tipo: **{pred}**")
    else:
        st.warning("⚠️ Por favor, escribe un titular.")
