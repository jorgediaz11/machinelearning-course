import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

# Configuración de la página
st.set_page_config(page_title="Predicción de Churn", page_icon="", layout="centered")

# Título y descripción
st.title("📉 Sistema de Predicción de Fuga de Clientes (Churn)")
st.markdown("Ingresa los datos del cliente para evaluar su probabilidad de cancelar el servicio.")

# --- DIAGNÓSTICO DE RUTAS (Para encontrar el error) ---
BASE_DIR = Path(__file__).parent.resolve()
MODELS_DIR = (BASE_DIR / '..' / 'models').resolve()

st.info(f"📍 La app está corriendo desde: `{BASE_DIR}`")
st.info(f"📍 Está buscando los modelos en: `{MODELS_DIR}`")

if not MODELS_DIR.exists():
    st.error(f" ¡La carpeta 'models' NO EXISTE en la ruta de arriba!")
    st.stop()
else:
    archivos_encontrados = os.listdir(MODELS_DIR)
    st.success(f"✅ Carpeta encontrada. Archivos dentro: {archivos_encontrados}")
    
    archivos_requeridos = ['xgb_final_model.joblib', 'scaler.joblib', 'train_columns.joblib']
    if not all(arch in archivos_encontrados for arch in archivos_requeridos):
        st.error(f"🚨 La carpeta existe, pero faltan archivos. Se requieren: {archivos_requeridos}")
        st.stop()

# --- CARGA DE ARTEFACTOS ---
@st.cache_resource
def cargar_artefactos():
    modelo = joblib.load(os.path.join(MODELS_DIR, 'xgb_final_model.joblib'))
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))
    columnas = joblib.load(os.path.join(MODELS_DIR, 'train_columns.joblib'))
    return modelo, scaler, columnas

modelo, scaler, columnas_entrenamiento = cargar_artefactos()

# --- FORMULARIO DE ENTRADA ---
st.subheader("Datos del Cliente")
col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Antigüedad (meses)", min_value=0, max_value=72, value=10)
    monthly_charges = st.number_input("Cargo Mensual ($)", min_value=0.0, max_value=120.0, value=65.0)
    contract = st.selectbox("Tipo de Contrato", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox("Método de Pago", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

with col2:
    internet_service = st.selectbox("Servicio de Internet", ["DSL", "Fiber optic", "No"])
    tech_support = st.selectbox("Soporte Técnico", ["No", "Yes"])
    online_security = st.selectbox("Seguridad en Línea", ["No", "Yes"])
    paperless_billing = st.selectbox("Facturación sin Papel", ["No", "Yes"])

# --- PROCESAMIENTO Y PREDICCIÓN ---
if st.button("🔮 Predecir Probabilidad de Fuga"):
    datos_cliente = pd.DataFrame({
        'tenure': [tenure],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [tenure * monthly_charges],
        'Contract_Month-to-month': [1 if contract == "Month-to-month" else 0],
        'Contract_One year': [1 if contract == "One year" else 0],
        'Contract_Two year': [1 if contract == "Two year" else 0],
        'PaymentMethod_Electronic check': [1 if payment_method == "Electronic check" else 0],
        'InternetService_Fiber optic': [1 if internet_service == "Fiber optic" else 0],
        'InternetService_DSL': [1 if internet_service == "DSL" else 0],
        'TechSupport_Yes': [1 if tech_support == "Yes" else 0],
        'OnlineSecurity_Yes': [1 if online_security == "Yes" else 0],
        'PaperlessBilling_Yes': [1 if paperless_billing == "Yes" else 0],
        'gender_Male': [0],
        'SeniorCitizen': [0],
        'Partner_Yes': [0],
        'Dependents_Yes': [0]
    })

    for col in columnas_entrenamiento:
        if col not in datos_cliente.columns:
            datos_cliente[col] = 0
            
    datos_cliente = datos_cliente[columnas_entrenamiento]

    cols_a_escalar = ['tenure', 'MonthlyCharges', 'TotalCharges']
    datos_cliente[cols_a_escalar] = scaler.transform(datos_cliente[cols_a_escalar])

    probabilidad = modelo.predict_proba(datos_cliente)[0][1]
    prediccion_clase = modelo.predict(datos_cliente)[0]

    st.markdown("---")
    if prediccion_clase == 1:
        st.error(f" **ALERTA DE FUGA**: Este cliente tiene una probabilidad del **{probabilidad:.1%}** de cancelar el servicio.")
        st.markdown("**Recomendación:** Asignar inmediatamente a un agente de retención.")
    else:
        st.success(f"✅ **CLIENTE ESTABLE**: Probabilidad de fuga del **{probabilidad:.1%}**.")
        st.markdown("**Recomendación:** Mantener engagement normal.")