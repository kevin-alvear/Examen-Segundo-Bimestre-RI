import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import faiss
import google.generativeai as genai

# ==========================================
# 1. RECURSOS NLTK (cache)
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
        st.warning(f"Advertencia NLTK: {e}")

download_nltk_resources()

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, CrossEncoder

# ==========================================
# 2. CONFIGURACIÓN DE PÁGINA
# ==========================================
st.set_page_config(
    page_title="arXiv RAG Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS profesionales + académicos
st.markdown("""
<style>
    /* Estilo general */
    .main {
        background-color: #f8f9fa;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    h1, h2, h3 {
        color: #1f2a3a;
    }
    /* Encabezado principal */
    .rag-header {
        background: linear-gradient(135deg, #0b1a2e 0%, #1a2f44 100%);
        padding: 1.8rem 2rem 1.2rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }
    .rag-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
        color: white;
    }
    .rag-header .subtitle {
        font-size: 0.95rem;
        margin: 0.2rem 0 0 0;
        opacity: 0.85;
        color: #d0dce8;
    }
    .rag-header .academic-info {
        font-size: 0.75rem;
        margin-top: 0.6rem;
        opacity: 0.6;
        color: #b0c9e0;
        border-top: 1px solid rgba(255,255,255,0.08);
        padding-top: 0.6rem;
        letter-spacing: 0.3px;
    }
    .rag-header .badge {
        display: inline-block;
        background: rgba(255,255,255,0.10);
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 500;
        margin-right: 0.4rem;
        color: #b0c9e0;
    }
    /* Tarjeta de estado en sidebar */
    .status-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 0.8rem 1rem;
        border: 1px solid #e9edf4;
        margin-top: 0.5rem;
    }
    .status-item {
        display: flex;
        justify-content: space-between;
        font-size: 0.85rem;
        padding: 0.2rem 0;
        border-bottom: 1px solid #f0f2f6;
    }
    .status-item:last-child {
        border-bottom: none;
    }
    .status-label {
        color: #5a6a7a;
    }
    .status-value {
        font-weight: 500;
        color: #0b1a2e;
    }
    /* Botones de sugerencia */
    .suggestion-btn {
        border-radius: 30px !important;
        border: 1px solid #d0d8e0 !important;
        background: white !important;
        color: #1f2a3a !important;
        font-weight: 500 !important;
        font-size: 0.8rem !important;
        padding: 0.3rem 0.8rem !important;
        transition: all 0.2s ease;
        box-shadow: 0 1px 4px rgba(0,0,0,0.02);
    }
    .suggestion-btn:hover {
        background: #eef2f7 !important;
        border-color: #a0b0c0 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    /* Footer académico */
    .footer {
        text-align: center;
        font-size: 0.7rem;
        color: #8a9aa8;
        padding: 1.2rem 0 0.5rem 0;
        border-top: 1px solid #e9edf4;
        margin-top: 2rem;
    }
    .footer span {
        margin: 0 0.6rem;
    }
    .footer .sep {
        color: #d0d8e0;
    }
    /* Ocultar elementos no deseados */
    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. ENCABEZADO CON INFORMACIÓN ACADÉMICA
# ==========================================
st.markdown("""
<div class="rag-header">
    <h1>📄 arXiv RAG Assistant</h1>
    <div class="subtitle">Búsqueda semántica y generación aumentada por recuperación sobre resúmenes de arXiv</div>
    <div style="margin-top:0.3rem;">
        <span class="badge">⚡ FAISS + Cross-Encoder</span>
        <span class="badge">🤖 Gemini</span>
    </div>
    <div class="academic-info">
        EPN · Facultad de Ingeniería de Sistemas · Recuperación de Información · Segundo Bimestre · Elaborado por Kevin Alvear
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 4. CARGA DE RECURSOS (cache)
# ==========================================
@st.cache_resource
def load_rag_resources():
    df = pd.read_csv('arxiv_corpus_processed.csv')
    corpus_embeddings = np.load('arxiv_embeddings.npy').astype('float32')
    dimension = corpus_embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(corpus_embeddings)
    bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return df, faiss_index, bi_encoder, cross_encoder

try:
    df, faiss_index, bi_encoder, cross_encoder = load_rag_resources()
except FileNotFoundError:
    st.error("❌ Error: archivos procesados no encontrados. Asegúrate de que 'arxiv_corpus_processed.csv' y 'arxiv_embeddings.npy' estén en el repositorio.")
    st.stop()

# ==========================================
# 5. CONFIGURACIÓN DE GEMINI
# ==========================================
if "gemini_configured" not in st.session_state:
    api_key = None
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    else:
        try:
            with open('gapi.txt', 'r') as f:
                api_key = f.read().strip()
        except FileNotFoundError:
            pass

    if api_key:
        genai.configure(api_key=api_key)
        st.session_state.gemini_model = genai.GenerativeModel('gemini-3.1-flash-lite')
        st.session_state.gemini_configured = True
    else:
        st.error("❌ API Key de Gemini no configurada. Configúrala en Secrets o en un archivo 'gapi.txt'.")
        st.stop()

# ==========================================
# 6. FUNCIONES AUXILIARES
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
# 7. BARRA LATERAL (configuración + estado)
# ==========================================
with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    k_retrieved = st.slider("Documentos a recuperar (K)", 3, 10, 5)
    use_reranking = st.checkbox("Activar re-ranking con Cross-Encoder", value=True)

    st.markdown("---")
    st.markdown("### 📊 Estado")
    st.markdown(f"""
    <div class="status-card">
        <div class="status-item"><span class="status-label">Abstracts</span><span class="status-value">{len(df)}</span></div>
        <div class="status-item"><span class="status-label">Índice</span><span class="status-value">FAISS L2</span></div>
        <div class="status-item"><span class="status-label">Re-ranking</span><span class="status-value">{'✅ Activo' if use_reranking else '❌ Inactivo'}</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("EPN · FIS · Recuperación de Información")
    st.caption("Kevin Alvear · Segundo Bimestre")

# ==========================================
# 8. ÁREA PRINCIPAL: SUGERENCIAS Y CHAT
# ==========================================
st.markdown("### 💬 Realizar consulta")
st.markdown("Escribe tu pregunta en el chat o prueba con estas sugerencias:")

col1, col2, col3, col4 = st.columns(4)
sugerencia_pulsada = ""

with col1:
    if st.button("📈 Graph Neural Networks", use_container_width=True, key="sug1"):
        sugerencia_pulsada = "What are the main applications of Graph Neural Networks?"
with col2:
    if st.button("🤖 RL in Robotics", use_container_width=True, key="sug2"):
        sugerencia_pulsada = "How is reinforcement learning used in robotics?"
with col3:
    if st.button("🎨 Diffusion Models", use_container_width=True, key="sug3"):
        sugerencia_pulsada = "Recent advances in diffusion models for image generation."
with col4:
    if st.button("⚡ Improving RAG", use_container_width=True, key="sug4"):
        sugerencia_pulsada = "Techniques for improving retrieval-augmented generation systems."

# Input del usuario
user_query = st.chat_input("Escribe tu consulta científica sobre arXiv...", key="chat_input")

# Determinar consulta final
if user_query:
    query_final = user_query
elif sugerencia_pulsada:
    query_final = sugerencia_pulsada
else:
    query_final = None

# ==========================================
# 9. PROCESAMIENTO DE LA CONSULTA (con limpieza)
# ==========================================
if query_final:
    # Limpiar historial anterior
    st.session_state.messages = []

    # Mostrar consulta del usuario
    with st.chat_message("user"):
        st.markdown(query_final)

    # Procesar RAG
    with st.chat_message("assistant"):
        with st.spinner("🔍 Recuperando documentos..."):
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
            with st.spinner("🔄 Reordenando con Cross-Encoder..."):
                pairs = [[query_final, cand["summary"]] for cand in candidates]
                rerank_scores = cross_encoder.predict(pairs)
                for i, score in enumerate(rerank_scores):
                    candidates[i]["rerank_score"] = float(score)
                candidates = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:k_retrieved]
        else:
            candidates = sorted(candidates, key=lambda x: x["score_l2"])[:k_retrieved]

        # Construir contexto
        context_parts = []
        for i, cand in enumerate(candidates):
            score_txt = f"Score: {cand['rerank_score']:.4f}" if use_reranking else f"Distancia L2: {cand['score_l2']:.4f}"
            context_parts.append(
                f"Documento {i+1} [{score_txt}]:\n"
                f"Título: {cand['title']}\n"
                f"Categorías: {cand['terms']}\n"
                f"Resumen: {cand['summary'][:800]}..."
            )
        context = "\n\n---\n\n".join(context_parts)

        # Prompt
        system_prompt = """
        Eres un asistente de investigación especializado en artículos de arXiv.
        Instrucciones:
        1. Responde únicamente usando la información del contexto proporcionado.
        2. Si el contexto no contiene información suficiente, di exactamente:
           "No tengo suficiente información en el corpus para responder esta pregunta."
        3. No inventes datos. Cita brevemente el título o categorías al responder.
        """
        user_prompt = f"Contexto:\n{context}\n\nPregunta:\n{query_final}\n\nRespuesta:"

        try:
            response = st.session_state.gemini_model.generate_content(f"{system_prompt}\n\n{user_prompt}")
            answer = response.text
        except Exception as e:
            answer = f"Error al generar respuesta: {str(e)}"

        # Mostrar respuesta
        st.markdown("### 📝 Respuesta")
        st.markdown(answer)

        # Trazabilidad
        st.markdown("---")
        st.markdown("### 🔬 Evidencias utilizadas")

        tab1, tab2 = st.tabs(["📄 Documentos", "📊 Tabla de similitud"])

        with tab1:
            for idx, cand in enumerate(candidates):
                with st.expander(f"Paper {idx+1}: {cand['title']}"):
                    st.write(f"**Categorías:** `{cand['terms']}`")
                    st.write(f"**Resumen:** {cand['summary']}")

        with tab2:
            table_data = []
            for idx, cand in enumerate(candidates):
                row = {
                    "Rank": idx+1,
                    "Título": cand["title"][:70] + "...",
                    "Distancia L2": round(cand["score_l2"], 4)
                }
                if use_reranking:
                    row["Score Cross-Encoder"] = round(cand["rerank_score"], 4)
                table_data.append(row)
            st.table(pd.DataFrame(table_data))

        # Guardar en sesión
        st.session_state.messages = [{"role": "user", "content": query_final},
                                     {"role": "assistant", "content": answer}]

else:
    st.info("👋 Realiza una consulta para comenzar.")

# ==========================================
# 10. FOOTER ACADÉMICO (al final de la página)
# ==========================================
st.markdown("""
<div class="footer">
    <span>Escuela Politécnica Nacional</span>
    <span class="sep">·</span>
    <span>Facultad de Ingeniería de Sistemas</span>
    <span class="sep">·</span>
    <span>Recuperación de Información</span>
    <span class="sep">·</span>
    <span>Segundo Bimestre 2026</span>
    <span class="sep">·</span>
    <span>Elaborado por Kevin Alvear</span>
</div>
""", unsafe_allow_html=True)