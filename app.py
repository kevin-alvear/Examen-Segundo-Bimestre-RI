import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import faiss
import google.generativeai as genai

# 1. Descargar y cachear de forma estrictamente segura todos los recursos de NLTK
# Se descarga 'punkt_tab' antes de inicializar cualquier procesamiento de texto
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

# Importar herramientas NLTK una vez garantizada la descarga
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer

# 2. Configuración de la interfaz
st.set_page_config(page_title="arXiv RAG Chat", page_icon="📚", layout="centered")
st.title("📚 Chat RAG - Búsqueda en arXiv")
st.write("Interfaz para consultar el corpus científico de arXiv de tu Examen.")

# 3. Cargar el corpus, los embeddings y el modelo usando caché para alto rendimiento
@st.cache_resource
def load_rag_resources():
    # Cargar dataframe
    df = pd.read_csv('arxiv_corpus_processed.csv')

    # Cargar matriz de embeddings
    corpus_embeddings = np.load('arxiv_embeddings.npy').astype('float32')

    # Crear índice FAISS
    dimension = corpus_embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(corpus_embeddings)

    # Cargar modelo de embeddings
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    return df, faiss_index, embedding_model

try:
    df, faiss_index, embedding_model = load_rag_resources()
except FileNotFoundError as e:
    st.error("❌ ERROR: No se encontraron los archivos procesados. Asegúrate de que 'arxiv_corpus_processed.csv' y 'arxiv_embeddings.npy' estén en la raíz de tu GitHub.")
    st.stop()

# 4. Configurar la API Key de Gemini (Nube con st.secrets o Local con gapi.txt)
if "gemini_configured" not in st.session_state:
    api_key = None

    # Intento 1: Buscar en los Secrets de Streamlit (Para la nube)
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]

    # Intento 2: Buscar en el archivo local gapi.txt (Para tu PC local)
    else:
        try:
            with open('gapi.txt', 'r') as file:
                api_key = file.read().strip()
        except FileNotFoundError:
            pass

    # Configurar el modelo si encontramos la clave
    if api_key:
        genai.configure(api_key=api_key)
        st.session_state.gemini_model = genai.GenerativeModel('gemini-3.1-flash-lite')
        st.session_state.gemini_configured = True
    else:
        st.error("ERROR: No se configuró la API Key de Gemini. Agrega 'GEMINI_API_KEY' en los Secrets de Streamlit (Nube) o crea un archivo 'gapi.txt' (Local).")
        st.stop()

# 5. Funciones auxiliares de tu cuaderno
def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(w) for w in tokens if w not in stop_words])

def search_documents(query, k=5):
    clean_query = clean_text(query)
    query_vector = embedding_model.encode([clean_query], convert_to_numpy=True).astype('float32')
    distances, indices = faiss_index.search(query_vector, k)
    # Aplanamos de forma explícita para evitar errores de dimensiones en Streamlit (TypeError)
    return np.array(distances[0]).flatten(), np.array(indices[0]).flatten()

# 6. Historial de Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada de texto del usuario
if query := st.chat_input("Escribe tu pregunta sobre los artículos..."):
    # Guardar y mostrar mensaje del usuario
    st.chat_message("user").markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # Buscar en base vectorial con FAISS de manera segura
    distances, indices = search_documents(query, k=5)

    # Construir contexto y evidencias de forma robusta
    context_parts = []
    evidences = []
    for i, idx in enumerate(indices):
        title = df.iloc[idx]['titles']
        summary = df.iloc[idx]['summaries']
        terms = df.iloc[idx]['terms']

        evidences.append(f"**{i+1}. {title}** (Distancia L2: {distances[i]:.4f})\nCategorías: {terms}\n\n*Resumen:* {summary}\n")
        context_parts.append(f"Documento {i+1}:\nTitulo: {title}\nResumen: {summary[:800]}...")

    context = "\n\n---\n\n".join(context_parts)

    # Prompts para el LLM (Idénticos a tu Jupyter Notebook)
    system_prompt = """
    Eres un asistente de investigacion especializado en articulos cientificos de arXiv.
    Instrucciones estrictas:
    1. Responde unicamente usando la informacion del contexto proporcionado.
    2. Si el contexto no contiene informacion suficiente para responder, debes decir:
       "No tengo suficiente informacion en el corpus para responder esta pregunta."
    3. No inventes datos, estadisticas o afirmaciones que no esten en el contexto.
    """
    user_prompt = f"Contexto:\n{context}\n\nPregunta:\n{query}\n\nRespuesta:"

    # Generar respuesta con Gemini
    with st.chat_message("assistant"):
        with st.spinner("Buscando en la base vectorial y generando respuesta..."):
            try:
                response = st.session_state.gemini_model.generate_content(f"{system_prompt}\n\n{user_prompt}")
                answer = response.text
            except Exception as e:
                answer = f"Error al generar respuesta con Gemini: {str(e)}"

            st.markdown(answer)

            # Mostrar evidencias abajo de la respuesta
            with st.expander("🔍 Ver fuentes/evidencias utilizadas"):
                for evidence in evidences:
                    st.markdown(evidence)
                    st.markdown("---")

        st.session_state.messages.append({"role": "assistant", "content": answer})