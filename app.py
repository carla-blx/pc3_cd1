# app.py
import streamlit as st
import time
from PIL import Image
import numpy as np
import pandas as pd
import torch

# Importar utilidades
from utils import (
    load_model, 
    preprocess_image, 
    load_class_names, 
    get_gradcam,
    get_top_predictions,
    format_confidence,
    CNN_DesdeCero  # Importamos también la clase por si acaso
)

# ============================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================
st.set_page_config(
    page_title="Clasificación de Enfermedades en Cultivos",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
/* Forzar colores de barras de progreso en todos los temas */
.stProgress > div > div {
    background-color: #e0e0e0 !important;
    border-radius: 10px !important;
}

/* Color de la barra de progreso (verde) */
.stProgress > div > div > div > div {
    background: #1F8FF2 !important;
    border-radius: 10px !important;
}

/* Color del texto del porcentaje */
.stProgress + div {
    color: #1b1b1b !important;
    font-weight: bold !important;
}

/* Para modo oscuro de Streamlit Cloud */
@media (prefers-color-scheme: dark) {
    .stProgress > div > div {
        background-color: #F5F5F5 !important;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #4caf50, #66bb6a) !important;
    }
    .stProgress + div {
        color: #ffffff !important;
    }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
:root {
    --primary: #2e7d32;
    --secondary: #66bb6a;
    --accent: #a5d6a7;
    --bg: #D3E8D4;
    --text: #1b1b1b;
}

.stApp {
    background-color: var(--bg);
    color: var(--text);
}

h1.main-header {
    color: var(--primary);
    text-align: center;
    font-weight: 800;
    margin-bottom: 10px;
    font-size: 2.5rem;
}

.prediction-box {
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    padding: 20px;
    border-radius: 12px;
    color: white;
    font-size: 18px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.2);
    margin-bottom: 20px;
}

.metric-card {
    background: white;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    text-align: center;
}

.stButton>button {
    background-color: var(--primary);
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 20px;
    font-weight: 600;
    transition: all 0.3s;
}

.stButton>button:hover {
    background-color: var(--secondary);
    color: black;
    transform: translateY(-2px);
}

.stAlert {
    border-radius: 10px;
}

.css-1v0mbdj, .css-1d391kg {
    background-color: white !important;
    border-radius: 10px;
    padding: 10px;
}

/* Barra de progreso personalizada */
.stProgress > div > div {
    background-color: var(--primary);
}
</style>
""", unsafe_allow_html=True)

# ============================================
# TÍTULO Y DESCRIPCIÓN
# ============================================
st.markdown("<h1 class='main-header'>🌱 Diagnóstico de Enfermedades en Hojas</h1>", 
            unsafe_allow_html=True)

st.markdown("""
    <div style='text-align: center; margin-bottom: 30px; font-size: 1.1rem;'>
    Sube una imagen de una hoja de cultivo y el modelo identificará automáticamente si está sana o 
    presenta alguna enfermedad común (PlantVillage dataset).
    </div>
""", unsafe_allow_html=True)

# ============================================
# CARGA DEL MODELO (con caché)
# ============================================
@st.cache_resource
def load_cached_model():
    """Carga el modelo y lo guarda en caché para reutilizar"""
    model_path = "modelo_cnn_scratch_final.pth"
    model, img_size = load_model(model_path)
    return model, img_size

# Cargar modelo y nombres de clases
try:
    with st.spinner("🔄 Cargando modelo... Esto puede tomar unos segundos"):
        model, IMG_SIZE = load_cached_model()
        class_names = load_class_names("class_names.txt")
except Exception as e:
    st.error(f"❌ Error al cargar el modelo: {str(e)}")
    st.info("Por favor, asegúrate de que el archivo 'modelo_cnn_scratch_final.pth' existe en el directorio actual.")
    st.stop()

# ============================================
# INTERFAZ PRINCIPAL
# ============================================
uploaded_file = st.file_uploader(
    "📤 Selecciona una imagen de hoja (JPG, PNG, JPEG)", 
    type=["jpg", "jpeg", "png"],
    help="Formatos aceptados: JPG, JPEG, PNG"
)

if uploaded_file is not None:
    # Cargar imagen
    image = Image.open(uploaded_file).convert('RGB')
    
    # Crear columnas
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        st.markdown("### 📷 Imagen Original")
        st.image(image, use_container_width=True)
        
        # Mostrar info de la imagen
        st.caption(f"Tamaño: {image.size[0]} x {image.size[1]} píxeles | Formato: {image.format}")
    
    # Preprocesamiento e inferencia
    with st.spinner("🔍 Analizando imagen..."):
        input_tensor = preprocess_image(image, size=IMG_SIZE)
        
        # Medir tiempo de inferencia
        start_time = time.time()
        with torch.no_grad():
            outputs = model(input_tensor.unsqueeze(0))
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            pred_idx = torch.argmax(probabilities).item()
            confidence = probabilities[pred_idx].item()
        inference_time = time.time() - start_time
        
        predicted_class = class_names[pred_idx] if pred_idx < len(class_names) else f"Clase_{pred_idx}"
    
    with col2:
        st.markdown("### 🔍 Resultado de la Predicción")
        
        # Mostrar predicción principal
        st.markdown(f"""
        <div class='prediction-box'>
        <span style='font-size: 24px; font-weight: bold;'>{predicted_class}</span><br><br>
        <b>🎯 Confianza:</b> {confidence:.2%}<br>
        <b>⏱️ Tiempo de inferencia:</b> {inference_time:.3f} segundos
        </div>
        """, unsafe_allow_html=True)
        

        # Top 5 predicciones
        st.markdown("### 📊 Top 5 Predicciones")
        top_predictions = get_top_predictions(probabilities, class_names, top_k=5)
        
        for class_name, prob in top_predictions:
            st.write(f"**{class_name}**")
            st.progress(float(prob))
            st.caption(f"{prob*100:.2f}%")
    
    # ============================================
    # GRAD-CAM VISUALIZATION
    # ============================================
    st.markdown("---")
    st.markdown("### Interpretabilidad Visual (Grad-CAM)")
    st.markdown("""
    <div style='font-size: 0.95rem; margin-bottom: 20px;'>
    El mapa de calor muestra qué regiones de la imagen fueron más importantes para la decisión del modelo.
    </div>
    """, unsafe_allow_html=True)
    
    # Seleccionar capa para Grad-CAM
    target_layer = model.features[-1].block[3]
    
    # Generar Grad-CAM
    with st.spinner("🎨 Generando mapa de activación..."):
        gradcam_image = get_gradcam(
            model, 
            input_tensor.unsqueeze(0), 
            target_layer, 
            target_class=pred_idx, 
            original_image=image,
            alpha=0.6
        )
    
    # Mostrar Grad-CAM centrado
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(gradcam_image, 
                caption="🔴 Áreas de mayor influencia para la predicción", 
                use_container_width=True,
                clamp=True)
    
    # Explicación
    with st.expander("¿Qué significa este mapa de calor?"):
        st.markdown("""
        - **Áreas rojas/amarillas**: Regiones que el modelo consideró más importantes para su decisión
        - **Áreas azules/verdes**: Regiones con menor influencia
        - **Interpretación**: Si el modelo predijo correctamente, las áreas resaltadas deben corresponder 
          a las partes de la hoja que muestran síntomas de la enfermedad
        - **Utilidad**: Ayuda a verificar si el modelo está usando características relevantes o si está 
          aprendiendo patrones espurios
        """)
    


else:
    # Mostrar información cuando no hay imagen
    st.info("👈 **Comienza subiendo una imagen** de una hoja de cultivo en el botón de arriba")
    
    # Mostrar ejemplo de interfaz
    with st.expander("📖 Ejemplos de imágenes que puedes subir"):
        st.markdown("""
        - Hojas de tomate con tizón temprano
        - Hojas de papa saludables
        - Hojas de manzana con costra
        - Hojas de maíz con roya
        - Cualquier hoja de cultivo agrícola
        """)

# ============================================
# PIE DE PÁGINA
# ============================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <b>Desarrollado con:</b> PyTorch, Streamlit, Grad-CAM<br>
</div>
""", unsafe_allow_html=True)
