# 📚 Sistema RAG - arXiv Abstracts

Este examen implementa un sistema completo de **Generación Aumentada por Recuperación (RAG)** diseñado para consultar y sintetizar información científica a partir de un corpus de 10,000 resúmenes de artículos académicos de **arXiv**. El sistema utiliza embeddings semánticos, almacenamiento e indexación vectorial eficiente con **FAISS** y el modelo **Gemini 3.1 Flash-Lite** para la generación de respuestas precisas, fieles y libres de alucinaciones.

## 📁 Estructura del Proyecto

Para asegurar que la aplicación web y el notebook funcionen correctamente, los archivos en el repositorio de GitHub deben estar organizados de la siguiente manera:

```text
/
│
├── data/
│   └── arxiv_data.csv             # Dataset original (opcional si es muy pesado)
│
├── arxiv_corpus_processed.csv     # Dataset preprocesado (generado por el notebook)
├── arxiv_embeddings.npy           # Embeddings vectoriales precalculados (.npy)
├── Alvear-Kevin-ExamenB2.ipynb    # Jupyter Notebook con el desarrollo paso a paso
├── app.py                         # Código de la interfaz interactiva de Streamlit
├── requirements.txt               # Archivo de dependencias para el despliegue
├── gapi.txt                       # Archivo local con la API Key de Gemini (¡No subir a GitHub!)
└── README.md                      # Documentación del proyecto (este archivo)