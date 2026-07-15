import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import faiss
import google.generativeai as genai

# ==========================================
# 1. DESCARGA SEGURA DE RECURSOS NLTK
# ==========================================
@st.cache_resource
def download_nltk_resources():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except Exception as e:
        st.warning(f"Advertencia al descargar recursos de NLTK: {e}")

download_nltk_resources()

# Importaciones adicionales de procesamiento
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, CrossEncoder

# ==========================================
# 2. CONFIGURACIÓN DE PÁGINA E INTERFAZ
# ==========================================
st.set_page_config(
    page_title="arXiv RAG System - EPN",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS inyectados usando un formato crudo limpio de Python (Raw String)
css_style = """
<style>
html, body, [class*='css'] {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.epn-banner {
    background: linear-gradient(135deg, #0A192F 0%, #172A45 100%);
    padding: 25px;
    border-radius: 12px;
    margin-bottom: 20px;
    color: white;
    border-left: 8px solid #D4AF37;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}
.epn-title {
    font-size: 26px !important;
    font-weight: 700 !important;
    margin: 0 !important;
    letter-spacing: 1px;
    color: #ffffff !important;
}
.epn-subtitle {
    font-size: 13px !important;
    margin: 5px 0 0 0 !important;
    color: #D4AF37 !important;
    font-weight: 600;
    text-transform: uppercase;
}
.credits-box {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 25px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    border-right: 6px solid #0A192F;
}
.credits-title {
    color: #0A192F;
    font-weight: 700;
    margin-bottom: 5px;
    font-size: 15px;
}
.stButton>button {
    border-radius: 8px !important;
    border: 1px solid #172A45 !important;
    background-color: #ffffff !important;
    color: #172A45 !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    padding: 6px 12px !important;
    transition: all 0.2s ease;
}
.stButton>button:hover {
    background-color: #172A45 !important;
    color: #D4AF37 !important;
    border-color: #D4AF37 !important;
}
</style>
"""
st.markdown(css_style, unsafe_allow_html=True)

# Cabecera Institucional
st.markdown(
    "<div class='epn-banner'>"
    "    <div class='epn-title'>ESCUELA POLITÉCNICA NACIONAL</div>"
    "    <div class='epn-subtitle'>Facultad de Ingeniería de Sistemas | Recuperación de Información</div>"
    "</div>",
    unsafe_allow_html=True
)

# Sección de Firma y Créditos Académicos (Kevin Alvear)
st.markdown(
    "<div class='credits-box'>"
    "    <table style='width:100%; border:none; border-collapse:collapse; background-color:transparent;'>"
    "        <tr style='border:none; background-color:transparent;'>"
    "            <td style='width:50%; border:none; padding:0; vertical-align:top; background-color:transparent;'>"
    "                <div class='credits-title'>🎓 EVALUACIÓN PRÁCTICA</div>"
    "                <span style='color:#4a5568; font-size:13px;'>"
    "                    <strong>Examen:</strong> Segundo Bimestre<br>"
    "                    <strong>Proyecto:</strong> RAG Pipeline de Dos Etapas (FAISS + Cross-Encoder)"
    "                </span>"
    "            </td>"
    "            <td style='width:50%; border:none; padding:0; text-align:right; vertical-align:top; background-color:transparent;'>"
    "                <div class='credits-title'>👤 AUTORÍA</div>"
    "                <span style='color:#4a5568; font-size:13px;'>"
    "                    <strong>Elaborado por:</strong> Kevin Xavier Alvear Cachipuendo<br>"
    "                    <strong>Docente:</strong> Dr. Iván Carrera"
    "                </span>"
    "            </td>"
    "        </tr>"
    "    </table>"
    "</div>",
    unsafe_html=True
)

# ==========================================
# 3. CARGA DE RECURSOS DEL EXAMEN (CACHÉ)
# ==========================================
@st.cache_resource
def load_rag_resources():
    # Cargar base de datos preprocesada
    df = pd.read_csv('arxiv_corpus_processed.csv')

    # Cargar embeddings
    corpus_embeddings = np.load('arxiv_embeddings.npy').astype('float32')

    # Inicializar base de datos vectorial FAISS L2
    dimension = corpus_embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(corpus_embeddings)

    # Modelos del flujo
    bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    return df, faiss_index, bi_encoder, cross_encoder

try:
    df, faiss_index, bi_encoder, cross_encoder = load_rag_resources()
except FileNotFoundError:
    st.error("❌ ERROR: No se encontraron los archivos procesados. Asegúrate de que 'arxiv_corpus_processed.csv' y 'arxiv_embeddings.npy' estén en tu GitHub.")
    st.stop()

# ==========================================
# 4. CONFIGURACIÓN DE GEMINI (NUBE & LOCAL)
# ==========================================
if "gemini_configured" not in st.session_state:
    api_key = None
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    else:
        try:
            with open('gapi.txt', 'r') as file:
                api_key = file.read().strip()
        except FileNotFoundError:
            pass

    if api_key:
        genai.configure(api_key=api_key)
        st.session_state.gemini_model = genai.GenerativeModel('gemini-3.1-flash-lite')
        st.session_state.gemini_configured = True
    else:
        st.error("❌ ERROR CRÍTICO: No se ha configurado la API Key de Gemini. Agrégala en los Secrets de la nube o en un archivo local 'gapi.txt'.")
        st.stop()

# ==========================================
# 5. FUNCIONES AUXILIARES DE PROCESAMIENTO
# ==========================================
def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(w) for w in tokens if w not in stop_words])

def search_documents(query, k=10):
    clean_query = clean_text(query)
    query_vector = bi_encoder.encode([clean_query], convert_to_numpy=True).astype('float32')
    distances, indices = faiss_index.search(query_vector, k)
    return np.array(distances[0]).flatten(), np.array(indices[0]).flatten()

# ==========================================
# 6. CONFIGURACIONES DE LA BARRA LATERAL
# ==========================================
st.sidebar.markdown(
    "<div style='text-align:center; padding-bottom:10px;'>"
    "    <h3 style='color:#0A192F; margin:0; font-weight:700;'>🛠️ CONFIGURACIÓN</h3>"
    "    <p style='font-size:12px; color:#64748B;'>Calibración del RAG</p>"
    "</div>",
    unsafe_html=True
)

k_retrieved = st.sidebar.slider("Documentos finales a recuperar (K)", min_value=3, max_value=10, value=5)
use_reranking = st.sidebar.checkbox("Activar Re-ranking (Cross-Encoder)", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Estado de Recursos")
st.sidebar.success(f"Base de Datos: {len(df)} abstracts")
st.sidebar.success("Índice FAISS L2: Activo")
if use_reranking:
    st.sidebar.success("Filtro Re-ranking: Activo")
else:
    st.sidebar.warning("Filtro Re-ranking: Inactivo")

# ==========================================
# 7. MANEJO DE SUGERENCIAS E INPUT DEL USUARIO
# ==========================================
st.write("### 💬 Realizar Consulta Semántica")
st.write("Escribe tu pregunta directamente en la barra de chat inferior, o haz clic en cualquiera de las consultas sugeridas del examen para autocompletar e iniciar la búsqueda:")

# Botones rápidos con consultas del examen
col1, col2, col3, col4 = st.columns(4)
sugerencia_pulsada = ""

with col1:
    if st.button("📈 Graph Neural Networks", use_container_width=True):
        sugerencia_pulsada = "What are the main applications of Graph Neural Networks?"
with col2:
    if st.button("🤖 RL in Robotics", use_container_width=True):
        sugerencia_pulsada = "How is reinforcement learning used in robotics?"
with col3:
    if st.button("🎨 Diffusion Models", use_container_width=True):
        sugerencia_pulsada = "Recent advances in diffusion models for image generation."
with col4:
    if st.button("⚡ Improving RAG", use_container_width=True):
        sugerencia_pulsada = "Techniques for improving retrieval-augmented generation systems."

# Variable de control en session_state para inyectar sugerencias de forma limpia
if "input_val" not in st.session_state:
    st.session_state.input_val = ""

if sugerencia_pulsada:
    st.session_state.input_val = sugerencia_pulsada

# Campo de entrada de texto
user_query = st.chat_input("Escribe tu consulta científica sobre arXiv aquí...", key="chat_input")

# Si el usuario hace clic en una sugerencia, la consulta a procesar será la de la sugerencia
query_final = user_query if user_query else st.session_state.input_val

# ==========================================
# 8. EJECUCIÓN DEL PIPELINE Y VISUALIZACIÓN
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Dibujar mensajes previos en pantalla
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Si el sistema recibe una consulta (sea manual o autocompletada)
if query_final:
    # 1. Limpiar la variable temporal para evitar bucles de renderizado
    if sugerencia_pulsada:
        st.session_state.input_val = ""

    # 2. Registrar y dibujar mensaje del usuario en pantalla
    st.chat_message("user").markdown(query_final)
    st.session_state.messages.append({"role": "user", "content": query_final})

    # 3. Procesamiento RAG de Dos Etapas en el bloque del Asistente
    with st.chat_message("assistant"):
        with st.spinner("1/2 Recuperando candidatos semánticos en FAISS L2..."):
            # Si hay Re-ranking, recuperamos el doble de candidatos para reordenarlos
            retrieve_k = k_retrieved * 2 if use_reranking else k_retrieved
            distances, indices = search_documents(query_final, k=retrieve_k)

            candidates = []
            for dist, idx in zip(distances, indices):
                candidates.append({
                    "title": df.iloc[idx]['titles'],
                    "summary": df.iloc[idx]['summaries'],
                    "terms": df.iloc[idx]['terms'],
                    "score_l2": float(dist)
                })

        if use_reranking:
            with st.spinner("2/2 Reordenando candidatos con Cross-Encoder..."):
                pairs = [[query_final, cand["summary"]] for cand in candidates]
                rerank_scores = cross_encoder.predict(pairs)
                for i, score in enumerate(rerank_scores):
                    candidates[i]["rerank_score"] = float(score)

                # Orden descendente (más alto es mejor)
                candidates = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:k_retrieved]
        else:
            # Orden ascendente (menor distancia vectorial es mejor)
            candidates = sorted(candidates, key=lambda x: x["score_l2"])[:k_retrieved]

        # Construir el contexto para Gemini
        context_parts = []
        for i, cand in enumerate(candidates):
            score_txt = f"Score Re-ranking: {cand['rerank_score']:.4f}" if use_reranking else f"Distancia L2: {cand['score_l2']:.4f}"
            context_parts.append(
                f"Documento {i+1} [{score_txt}]:\n"
                f"Título: {cand['title']}\n"
                f"Categorías: {cand['terms']}\n"
                f"Resumen: {cand['summary'][:800]}..."
            )
        context = "\n\n---\n\n".join(context_parts)

        # Prompt restrictivo para evitar alucinaciones
        system_prompt = """
        Eres un asistente de investigacion de la Escuela Politecnica Nacional especializado en arXiv.
        Instrucciones estrictas:
        1. Responde unicamente usando la informacion del contexto proporcionado de manera muy formal y resumida.
        2. Si el contexto no contiene informacion suficiente para responder, debes decir exactamente:
           "No tengo suficiente informacion en el corpus para responder esta pregunta."
        3. No asumas ni inventes datos cientificos. Cita brevemente el autor o titulo al responder.
        """
        user_prompt = f"Contexto de evidencias:\n{context}\n\nPregunta:\n{query_final}\n\nRespuesta:"

        # Generar respuesta
        try:
            response = st.session_state.gemini_model.generate_content(f"{system_prompt}\n\n{user_prompt}")
            answer = response.text
        except Exception as e:
            answer = f"Error al generar la respuesta semántica: {str(e)}"

        # Imprimir respuesta
        st.markdown("### 📝 Respuesta Generada (RAG)")
        st.markdown(answer)

        # 4. Mostrar Trazabilidad e Información de Recuperación
        st.markdown("---")
        st.markdown("### 🔬 Trazabilidad y Evidencias Semánticas")

        tab1, tab2 = st.tabs(["📄 Artículos del Contexto", "📊 Tabla de Scores y Similitud"])

        with tab1:
            for idx, cand in enumerate(candidates):
                with st.expander(f"Paper {idx+1}: {cand['title']}"):
                    st.write(f"**Categorías:** `{cand['terms']}`")
                    st.write(f"**Resumen completo:** {cand['summary']}")

        with tab2:
            st.write("Tabla comparativa de scores de búsqueda inicial (FAISS) y reordenamiento semántico (Cross-Encoder):")
            table_data = []
            for idx, cand in enumerate(candidates):
                row = {
                    "Ranking": idx + 1,
                    "Título": cand["title"][:70] + "...",
                    "Distancia L2 (FAISS)": round(cand["score_l2"], 4)
                }
                if use_reranking:
                    row["Score Re-ranking (Cross-Encoder)"] = round(cand["rerank_score"], 4)
                table_data.append(row)

            st.table(pd.DataFrame(table_data))

        # Registrar la respuesta del asistente
        st.session_state.messages.append({"role": "assistant", "content": answer})