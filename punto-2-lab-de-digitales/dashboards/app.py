import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Prevención del STC", layout="wide")

# Cargar datos procesados
df = pd.read_csv("data/processed/dataset_analitico.csv")

st.title("Sistema de prevención del síndrome de túnel carpiano")

# Métricas principales
col1, col2, col3 = st.columns(3)
col1.metric("Pacientes", len(df["id_paciente"].unique()))
col2.metric("Evaluaciones", len(df))
col3.metric("Riesgo alto (%)", f"{(df['riesgo_label']=='alto').mean()*100:.1f}%")

# Distribución de riesgo
st.subheader("Distribución de riesgo")
fig_risk = px.histogram(df, x="riesgo_score", color="riesgo_label", nbins=30)
st.plotly_chart(fig_risk, use_container_width=True)

# Relación entre edad y riesgo
st.subheader("Edad vs Riesgo")
fig_age = px.scatter(df, x="edad", y="riesgo_score", color="riesgo_label",
                     hover_data=["nombre", "ciudad"])
st.plotly_chart(fig_age, use_container_width=True)

# Relación entre fuerza y latencia EMG
st.subheader("Fuerza vs Latencia EMG")
fig_emg = px.scatter(df, x="fuerza_prension_kg", y="latencia_emg_ms",
                     color="riesgo_label", hover_data=["nombre"])
st.plotly_chart(fig_emg, use_container_width=True)
