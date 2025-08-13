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
        "Proyecto hidroeléctrico reduce uso de comb
