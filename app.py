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

# Importación de herramientas una vez garantizadas las descargas
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

# Estilo CSS Personalizado - Paleta de colores Premium y Académica (Azul Marino y Dorado)
st.markdown("""
    <style>
        /* Tipografía y fondo general */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Banner Principal de la Escuela Politécnica Nacional */
        .epn-banner {
            background: linear-gradient(135deg, #0A192F 0%, #172A45 100%);
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 20px;
            color: white;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
            border-left: 8px solid #D4AF37; /* Dorado */
            text-align: center;
        }
        .epn-title {
            font-size: 28px !important;
            font-weight: 700 !important;
            margin: 0 !important;
            letter-spacing: 1px;
            color: #ffffff !important;
        }
        .epn-subtitle {
            font-size: 14px !important;
            margin: 6px 0 0 0 !important;
            color: #D4AF37 !important;
            font-weight: 600;
            text-transform: uppercase;
        }

        /* Tarjeta de Créditos de Autor */
        .credits-box {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            padding: 18px;
            border-radius: 10px;
            margin-bottom: 25px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            border-right: 6px solid #0A192F;
        }
        .credits-title {
            color: #0A192F;
            font-weight: 700;
            margin-bottom: 5px;
            font-size: 16px;
        }

        /* Estilos de botones de sugerencias */
        .stButton>button {
            border-radius: 8px !important;
            border: 1px solid #172A45 !important;
            background-color: #ffffff !important;
            color: #172A45 !important;
            font-weight: 500 !important;
            font-size: 13px !important;
            padding: 8px 12px !important;
            transition: all 0.2s ease;
        }
        .stButton>button:hover {
            background-color: #172A45 !important;
            color: #D4AF37 !important;
            border-color: #D4AF37 !important;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_html=True)

# 1. Cabecera Institucional
st.markdown("""
    <div class="epn-banner">
        <div class="epn-title">ESCUELA POLITÉCNICA NACIONAL</div>
        <div class="epn-subtitle">Facultad de Ingeniería de Sistemas | Recuperación de Información</div>
    </div>
""", unsafe_html=True)

# 2. Información del Estudiante y Examen (Súper elegante y formal)
st.markdown("""
    <div class="credits-box">
        <table style="width:100%; border:none; border-collapse:collapse;">
            <tr style="border:none; background-color:transparent;">
                <td style="width:50%; border:none; padding:0; vertical-align:top;">
                    <div class="credits-title">🎓 EVALUACIÓN ACADÉMICA</div>
                    <span style="color:#4a5568; font-size:14px;">
                        <strong>Examen:</strong> Segundo Bimestre<br>
                        <strong>Tema:</strong> Arquitectura RAG de Dos Etapas (FAISS + Cross-Encoder)
                    </span>
                </td>
                <td style="width:50%; border:none; padding:0; text-align:right; vertical-align:top;">
                    <div class="credits-title">👤 AUTORÍA</div>
                    <span style="color:#4a5568; font-size:14px;">
                        <strong>Elaborado por:</strong> Kevin Xavier Alvear Cachipuendo<br>
                        <strong>Docente:</strong> Dr. Iván Carrera
                    </span>
                </td>
            </tr>
        </table>
    </div>
""", unsafe_html=True)

# ==========================================
# 3. CARGA DE RECURSOS DEL EXAMEN (CACHÉ)
# ==========================================
@st.cache_resource
def load_rag_resources():
    # Carga de datos procesados
    df = pd.read_csv('arxiv_corpus_processed.csv')

    # Carga de Embeddings de arXiv
    corpus_embeddings = np.load('arxiv_embeddings.npy').astype('float32')

    # Inicialización del Índice FAISS L2
    dimension = corpus_embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(corpus_embeddings)

    # Modelos del Pipeline RAG
    bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    return df, faiss_index, bi_encoder, cross_encoder

try:
    df, faiss_index, bi_encoder, cross_encoder = load_rag_resources()
except FileNotFoundError:
    st.error("❌ ERROR: No se encontraron los archivos procesados. Sube 'arxiv_corpus_processed.csv' y 'arxiv_embeddings.npy' a tu GitHub.")
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
        st.error("❌ ERROR: API Key de Gemini no configurada. Añádela en los Secrets de Streamlit o localmente en 'gapi.txt'.")
        st.stop()

# ==========================================
# 5. FUNCIONES DE PROCESAMIENTO
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
st.sidebar.markdown("""
    <div style="text-align:center; padding-bottom:10px;">
        <h3 style="color:#0A192F; margin:0; font-weight:700;">🛠️ CONFIGURACIÓN</h3>
        <p style="font-size:12px; color:#64748B;">Ajuste de Parámetros RAG</p>
    </div>
""", unsafe_html=True)

k_retrieved = st.sidebar.slider("Documentos finales (K)", min_value=3, max_value=10, value=5)
use_reranking = st.sidebar.checkbox("Activar Re-ranking (Cross-Encoder)", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Estado del Pipeline")
st.sidebar.success(f"Base de Datos: {len(df)} Registros")
st.sidebar.success("Buscador FAISS: Listo")
st.sidebar.success("Generador Gemini: Listo")

# ==========================================
# 7. SECCIÓN DE SUGERENCIAS Y ENTRADA DE TEXTO
# ==========================================
st.write("### 💬 Realiza tu consulta")
st.write("Escribe tu propia pregunta en la barra inferior o haz clic en cualquiera de las consultas sugeridas del examen:")

# Botones rápidos con las preguntas exactas del PDF
col1, col2, col3, col4 = st.columns(4)
query_to_execute = ""

with col1:
    if st.button("📈 Graph Neural Networks", use_container_width=True):
        query_to_execute = "What are the main applications of Graph Neural Networks?"
with col2:
    if st.button("🤖 RL in Robotics", use_container_width=True):
        query_to_execute = "How is reinforcement learning used in robotics?"
with col3:
    if st.button("🎨 Diffusion Models", use_container_width=True):
        query_to_execute = "Recent advances in diffusion models for image generation."
with col4:
    if st.button("⚡ Improving RAG", use_container_width=True):
        query_to_execute = "Techniques for improving retrieval-augmented generation systems."

# Entrada de texto personalizada del usuario
user_text_input = st.chat_input("Escribe tu consulta científica sobre arXiv aquí...")

# La consulta final será la que venga de la barra de texto o la del botón pulsado
final_query = user_text_input if user_text_input else query_to_execute

# ==========================================
# 8. EJECUCIÓN Y RENDERIZADO DEL CHAT
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar el historial acumulado en la pantalla
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Si hay una consulta activa (sea escrita o pulsada en un botón)
if final_query:
    # 1. Registrar y pintar la pregunta del usuario
    st.chat_message("user").markdown(final_query)
    st.session_state.messages.append({"role": "user", "content": final_query})

    # 2. Procesar y generar respuesta
    with st.chat_message("assistant"):
        with st.spinner("Recuperando documentos semánticos en FAISS..."):
            # Si se usa Re-ranking recuperamos el doble de candidatos para reordenar
            retrieve_k = k_retrieved * 2 if use_reranking else k_retrieved
            distances, indices = search_documents(final_query, k=retrieve_k)

            candidates = []
            for dist, idx in zip(distances, indices):
                candidates.append({
                    "title": df.iloc[idx]['titles'],
                    "summary": df.iloc[idx]['summaries'],
                    "terms": df.iloc[idx]['terms'],
                    "score_l2": float(dist)
                })

        # Paso de Re-ranking de dos etapas (Requerimiento Clave)
        if use_reranking:
            with st.spinner("Reordenando resultados con Cross-Encoder..."):
                pairs = [[final_query, cand["summary"]] for cand in candidates]
                rerank_scores = cross_encoder.predict(pairs)
                for i, score in enumerate(rerank_scores):
                    candidates[i]["rerank_score"] = float(score)

                # Ordenar descendente (mayor puntaje del Cross-Encoder es mejor)
                candidates = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:k_retrieved]
        else:
            # Ordenar ascendente (menor distancia vectorial es mejor)
            candidates = sorted(candidates, key=lambda x: x["score_l2"])[:k_retrieved]

        # Crear el contexto unificado para el LLM
        context_parts = []
        for i, cand in enumerate(candidates):
            score_label = f"Score Re-ranking: {cand['rerank_score']:.4f}" if use_reranking else f"Distancia L2: {cand['score_l2']:.4f}"
            context_parts.append(
                f"Documento {i+1} [{score_label}]:\n"
                f"Título: {cand['title']}\n"
                f"Categorías: {cand['terms']}\n"
                f"Resumen: {cand['summary'][:800]}..."
            )
        context = "\n\n---\n\n".join(context_parts)

        # Prompt de sistema restrictivo para evitar alucinaciones (Rúbrica de evaluación)
        system_prompt = """
        Eres un asistente de investigacion de la Escuela Politecnica Nacional especializado en arXiv.
        Instrucciones estrictas:
        1. Responde unicamente usando la informacion del contexto proporcionado de manera muy formal y resumida.
        2. Si el contexto no contiene informacion suficiente para responder, debes decir textualmente:
           "No tengo suficiente informacion en el corpus para responder esta pregunta."
        3. No asumas ni inventes datos cientificos. Cita brevemente el autor o titulo al responder.
        """
        user_prompt = f"Contexto de evidencias:\n{context}\n\nPregunta:\n{final_query}\n\nRespuesta:"

        # Invocar a Gemini
        try:
            response = st.session_state.gemini_model.generate_content(f"{system_prompt}\n\n{user_prompt}")
            answer = response.text
        except Exception as e:
            answer = f"Error al generar respuesta: {str(e)}"

        # Pintar la respuesta del modelo
        st.markdown("### 📝 Respuesta Generada (RAG)")
        st.markdown(answer)

        # 3. Mostrar la Trazabilidad y Métricas de las Evidencias (Rúbrica de Recuperación)
        st.markdown("---")
        st.markdown("### 🔬 Trazabilidad y Evidencias Semánticas")

        tab1, tab2 = st.tabs(["📄 Artículos del Contexto", "📊 Tabla de Scores y Similitud"])

        with tab1:
            for idx, cand in enumerate(candidates):
                with st.expander(f"Paper {idx+1}: {cand['title']}"):
                    st.write(f"**Categorías:** `{cand['terms']}`")
                    st.write(f"**Resumen completo:** {cand['summary']}")

        with tab2:
            st.write("Tabla comparativa de scores de búsqueda inicial (FAISS) y re-ranking (Cross-Encoder):")
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

        # Registrar la respuesta del asistente en el historial
        st.session_state.messages.append({"role": "assistant", "content": answer})