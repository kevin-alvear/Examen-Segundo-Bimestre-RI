# Sistema de Recuperación Aumentada por Generación (RAG) sobre Abstracts de arXiv

**Asignatura:** Recuperación de Información

**Tipo:** Examen Final

**Estudiante:** Kevin Xavier Alvear Cachipuendo

**Profesor:** Dr. Iván Carrera

**Fecha:** 15 de julio de 2026

---

## Descripción

Este proyecto implementa un sistema RAG (Retrieval-Augmented Generation) que responde consultas en lenguaje natural sobre un corpus de 25,000 resúmenes de artículos de arXiv. Combina búsqueda semántica con embeddings (`all-MiniLM-L6-v2`), indexación FAISS, re-ranking opcional con Cross-Encoder y generación de respuestas mediante Google Gemini (`gemini-3.1-flash-lite`). La aplicación se despliega como una interfaz web conversacional desarrollada con Streamlit.

---

## Características Principales

- Búsqueda semántica por similitud vectorial (FAISS L2).
- Re-ranking opcional con Cross-Encoder para mejorar la relevancia.
- Generación de respuestas contextualizadas con Gemini.
- Interfaz web tipo chat con trazabilidad de evidencias (documentos, métricas de similitud).
- Detección explícita de falta de información en el corpus.

---

## Arquitectura del Sistema

1. **Preparación del corpus:** limpieza, tokenización, lematización y combinación de título + resumen.
2. **Representación vectorial:** embeddings de dimensión 384 con `all-MiniLM-L6-v2` y almacenamiento en índice FAISS.
3. **Recuperación y re-ranking:** búsqueda inicial en FAISS, reordenamiento opcional con Cross-Encoder.
4. **Generación de respuesta:** construcción de contexto y llamada a Gemini con prompt restrictivo.
5. **Interfaz de usuario:** Streamlit con chat, configuración (K, re-ranking) y visualización de evidencias.

---

## Tecnologías Utilizadas

| Componente | Tecnología |
|------------|------------|
| Lenguaje | Python 3.13 |
| Procesamiento de datos | Pandas, NumPy |
| NLP | NLTK |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Base vectorial | FAISS (`IndexFlatL2`) |
| Re-ranking | Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) |
| Modelo generativo | Google Gemini (`gemini-3.1-flash-lite`) |
| Interfaz web | Streamlit |
| Despliegue | Streamlit Community Cloud |

---

## Instalación y Ejecución Local

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/tu-usuario/examen-segundo-bimestre-ri.git
   cd examen-segundo-bimestre-ri
   ```
2. Crear y activar un entorno virtual (opcional).
3. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
4. Configurar la clave de API de Gemini en un archivo `gapi.txt` o mediante Secrets de Streamlit.
5. Ejecutar la aplicación:
   ```bash
   streamlit run app.py
   ```

---

## Despliegue en la Nube

La aplicación está disponible en:
[https://kevin-alvear-examen-rag.streamlit.app/](https://kevin-alvear-examen-rag.streamlit.app/)

---

## Evaluación

El sistema fue evaluado cualitativamente sobre cinco consultas representativas, considerando:
- Corrección factual.
- Relevancia.
- Fidelidad al contexto.
- Capacidad de síntesis.
- Reconocimiento de insuficiencia de información.

Los resultados confirman que el sistema genera respuestas coherentes cuando existe evidencia suficiente y rechaza adecuadamente consultas fuera del dominio del corpus.

---

## Autor

**Kevin Xavier Alvear Cachipuendo**
Escuela Politécnica Nacional
Ingeniería en Ciencias de la Computación

---

## Licencia

Proyecto académico desarrollado para la asignatura Recuperación de Información.
```

---
