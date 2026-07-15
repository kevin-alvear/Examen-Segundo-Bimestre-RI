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

# Estilos CSS modernos y profesionales (sin elementos de créditos ni proyectos)
css_style = """
<style>
    /* Reset y fuente */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
        background-color: #f8fafc;
    }
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 2rem 2rem 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
        text-align: center;
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 600;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 1rem;
        margin: 0.3rem 0 0 0;
        font-weight: 400;
    }
    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #0f172a;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
        border-bottom: 3px solid #e2e8f0;
        padding-bottom: 0.3rem;
    }
    .suggestion-btn {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 30px;
        padding: 0.4rem 1.2rem;
        font-size: 0.85rem;
        font-weight: 500;
        color: #1e293b;
        transition: all 0.15s ease;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        width: 100%;
        text-align: center;
    }
    .suggestion-btn:hover {
        background-color: #0f172a;
        color: #ffffff;
        border-color: #0f172a;
        cursor: pointer;
    }
    /* Sidebar más limpio */
    .css-1d391kg, .css-1d391kg p {
        font-size: 0.9rem;
    }
    .sidebar-status {
        background-color: #f1f5f9;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
    }
    .stButton>button {
        border-radius: 30px !important;
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        color: #1e293b !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        padding: 0.4rem 1.2rem !important;
        transition: 0.15s;
    }
    .stButton>button:hover {
        background: #0f172a !important;
        color: #ffffff !important;
        border-color: #0f172a !important;
    }
    .chat-message {
        background: #ffffff;
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #e2e8f0;
        margin-bottom: 0.75rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.02);
    }
    .footer {
        margin-top: 3rem;
        text-align: center;
        color: #94a3b8;
        font-size: 0.75rem;
        border-top: 1px solid #e2e8f0;
        padding-top: 1.5rem;
    }
    .badge {
        background: #e2e8f0;
        padding: 0.2rem 0.7rem;
        border-radius: 30px;
        font-size: 0.7rem;
        font-weight: 600;
        color: #1e293b;
        display: inline-block;
        margin-right: 0.3rem;
    }
</style>
"""
st.markdown(css_style, unsafe_allow_html=True)

# ==========================================
# 3. CABECERA PRINCIPAL (sin autoria ni proyecto)
# ==========================================
st.markdown("""
<div class="main-header">
    <h1>📄 arXiv RAG Assistant</h1>
    <p>Búsqueda semántica y generación aumentada por recuperación sobre resúmenes de arXiv</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 4. CARGA DE RECURSOS (CACHÉ)
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
    st.error("❌ No se encontraron los archivos procesados. Asegúrate de que 'arxiv_corpus_processed.csv' y 'arxiv_embeddings.npy' estén disponibles.")
    st.stop()

# ==========================================
# 5. CONFIGURACIÓN DE GEMINI
# ==========================================
if "gemini_configured" not in st.session_state:
    api_key = st.secrets.get("GEMINI_API_KEY", None)
    if not api_key:
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
        st.error("❌ API Key de Gemini no configurada. Agrega 'GEMINI_API_KEY' en los secrets o en 'gapi.txt'.")
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
# 7. BARRA LATERAL (configuración)
# ==========================================
with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    k_retrieved = st.slider("Documentos a recuperar (K)", min_value=3, max_value=10, value=5)
    use_reranking = st.checkbox("Activar re‑ranking con Cross‑Encoder", value=True)

    st.markdown("---")
    st.markdown("### 📊 Estado")
    st.markdown(f"<div class='sidebar-status'>📚 {len(df)} resúmenes</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-status'>🔍 FAISS L2 activo</div>", unsafe_allow_html=True)
    if use_reranking:
        st.markdown("<div class='sidebar-status'>⚡ Re‑ranking activo</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='sidebar-status'>⏳ Re‑ranking inactivo</div>", unsafe_allow_html=True)

# ==========================================
# 8. ÁREA PRINCIPAL: CONSULTAS SUGERIDAS
# ==========================================
st.markdown('<div class="section-title">💬 Consulta semántica</div>', unsafe_allow_html=True)
st.markdown("Escribe tu pregunta en el chat o prueba con estas sugerencias:")

cols = st.columns(4)
sugerencias = {
    "📈 Graph Neural Networks": "What are the main applications of Graph Neural Networks?",
    "🤖 RL in Robotics": "How is reinforcement learning used in robotics?",
    "🎨 Diffusion Models": "Recent advances in diffusion models for image generation.",
    "⚡ Improving RAG": "Techniques for improving retrieval-augmented generation systems."
}

sugerencia_pulsada = ""
for col, (label, query) in zip(cols, sugerencias.items()):
    if col.button(label, use_container_width=True, key=label):
        sugerencia_pulsada = query

# ==========================================
# 9. CHAT INPUT Y PROCESAMIENTO
# ==========================================
if "input_val" not in st.session_state:
    st.session_state.input_val = ""

if sugerencia_pulsada:
    st.session_state.input_val = sugerencia_pulsada

user_query = st.chat_input("Escribe tu consulta sobre arXiv...", key="chat_input")
query_final = user_query if user_query else st.session_state.input_val

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query_final:
    # Limpiar sugerencia para evitar repetición
    if sugerencia_pulsada:
        st.session_state.input_val = ""

    st.chat_message("user").markdown(query_final)
    st.session_state.messages.append({"role": "user", "content": query_final})

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
            with st.spinner("⚡ Reordenando con Cross‑Encoder..."):
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
            score_txt = f"Score: {cand['rerank_score']:.4f}" if use_reranking else f"Distancia: {cand['score_l2']:.4f}"
            context_parts.append(
                f"Documento {i+1} [{score_txt}]:\n"
                f"Título: {cand['title']}\n"
                f"Categorías: {cand['terms']}\n"
                f"Resumen: {cand['summary'][:800]}..."
            )
        context = "\n\n---\n\n".join(context_parts)

        system_prompt = """
        Eres un asistente de investigación especializado en artículos de arXiv.
        Instrucciones:
        1. Responde únicamente usando la información del contexto proporcionado.
        2. Si el contexto no contiene información suficiente, responde exactamente:
           "No tengo suficiente información en el corpus para responder esta pregunta."
        3. No inventes datos ni citas. Sé conciso y formal.
        """
        user_prompt = f"Contexto:\n{context}\n\nPregunta:\n{query_final}\n\nRespuesta:"

        try:
            response = st.session_state.gemini_model.generate_content(f"{system_prompt}\n\n{user_prompt}")
            answer = response.text
        except Exception as e:
            answer = f"Error: {str(e)}"

        st.markdown("### 📝 Respuesta")
        st.markdown(answer)

        # Mostrar evidencias en pestañas
        st.markdown("---")
        st.markdown("### 📚 Evidencias utilizadas")
        tab1, tab2 = st.tabs(["📄 Artículos", "📊 Scores"])

        with tab1:
            for idx, cand in enumerate(candidates):
                with st.expander(f"{idx+1}. {cand['title']}"):
                    st.write(f"**Categorías:** `{cand['terms']}`")
                    st.write(f"**Resumen:** {cand['summary']}")

        with tab2:
            df_scores = pd.DataFrame([{
                "Rank": i+1,
                "Título": c["title"][:60] + "...",
                "Distancia L2": round(c["score_l2"], 4),
                "Rerank Score": round(c.get("rerank_score", 0), 4) if use_reranking else "-"
            } for i, c in enumerate(candidates)])
            st.dataframe(df_scores, use_container_width=True, hide_index=True)

        st.session_state.messages.append({"role": "assistant", "content": answer})

# ==========================================
# 10. PIE DE PÁGINA (opcional, sin créditos)
# ==========================================
st.markdown('<div class="footer">arXiv RAG Assistant · Recuperación de Información</div>', unsafe_allow_html=True)