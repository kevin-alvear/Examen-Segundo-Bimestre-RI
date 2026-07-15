import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import faiss
import google.generativeai as genai
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer

# 1. Configuración de la interfaz
st.set_page_config(page_title="arXiv RAG Chat", page_icon="📚", layout="centered")
st.title("📚 Chat RAG - Búsqueda en arXiv")
st.write("Elaborado por: Kevin Alvear.")
st.write("Interfaz local para consultar el corpus de arXiv.")

# 2. Descargar y cachear recursos de NLTK de forma segura
@st.cache_resource
def download_nltk_resources():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

download_nltk_resources()

# 3. Cargar el corpus, los embeddings y el modelo (usando caché para evitar lentitud)
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
    st.error("ERROR: No se encontraron los archivos procesados. Ejecuta tu notebook primero para generar 'arxiv_corpus_processed.csv' y 'arxiv_embeddings.npy'.")
    st.stop()

# 4. Configurar la API Key de Gemini (Compatible con local y nube)
if "gemini_configured" not in st.session_state:
    api_key = None

    # Intento 1: Buscar en los Secretos de Streamlit (Para la nube)
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]

    # Intento 2: Buscar en el archivo local gapi.txt (Para tu PC local)
    else:
        try:
            with open('gapi.txt', 'r') as file:
                api_key = file.read().strip()
        except FileNotFoundError:
            pass

    # Configurar el modelo si encontramos la clave en algún lado
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
    return distances[0], indices[0]

# 6. Lógica de renderizado del historial de Chat interactivo
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

    # Buscar en base vectorial
    distances, indices = search_documents(query, k=5)

    # Construir contexto
    context_parts = []
    evidences = []
    for i, idx in enumerate(indices):
        title = df.iloc[idx]['titles']
        summary = df.iloc[idx]['summaries']
        terms = df.iloc[idx]['terms']

        evidences.append(f"**{i+1}. {title}** (Distancia: {distances[i]:.4f})\nCategorías: {terms}\n\n*Resumen:* {summary}\n")
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
                answer = f"Error al generar respuesta: {str(e)}"

            st.markdown(answer)

            # Mostrar evidencias de forma limpia abajo de la respuesta
            with st.expander("🔍 Ver fuentes/evidencias utilizadas"):
                for evidence in evidences:
                    st.markdown(evidence)
                    st.markdown("---")

        st.session_state.messages.append({"role": "assistant", "content": answer})