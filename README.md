# 📉 Sistema de Predicción de Fuga de Clientes (Customer Churn)

## 📌 Descripción del Proyecto
Este proyecto desarrolla un modelo de Machine Learning para predecir la probabilidad de que un cliente de una empresa de telecomunicaciones cancele su servicio (Churn). El objetivo principal no es solo lograr una alta precisión, sino **entender los factores de riesgo** y proporcionar una herramienta interactiva para que el equipo de retención pueda actuar de manera preventiva.

## 🛠️ Tecnologías y Herramientas
*   **Lenguaje:** Python 3.10+
*   **Análisis de Datos:** Pandas, NumPy
*   **Visualización:** Matplotlib, Seaborn
*   **Machine Learning:** Scikit-learn, XGBoost, LightGBM
*   **Interpretabilidad:** SHAP (SHapley Additive exPlanations)
*   **Optimización:** Optuna
*   **Despliegue:** Streamlit

## 📂 Estructura del Proyecto
```text
churn-project/
├── data/               # Dataset original (Telco Customer Churn)
├── notebooks/          # Análisis exploratorio, modelado y evaluación
├── src/                # Código fuente de la aplicación Streamlit (app.py)
├── models/             # Modelos serializados y artefactos (scaler, columnas)
── .gitignore          # Archivos ignorados por Git
├── requirements.txt    # Dependencias del proyecto
└── README.md           # Documentación actual