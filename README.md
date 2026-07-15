# Sistema RAG - arXiv Paper Abstracts

## Examen Final - Recuperación de Información

**Autor:** Kevin Alvear
**Asignatura:** ICCD753 - Recuperación de Información
**Profesor:** Iván Carrera
**Fecha:** 15 de Julio 2026

---

## 🚀 Descripción del Proyecto

Sistema RAG (*Retrieval-Augmented Generation*) diseñado para responder preguntas sobre artículos científicos de arXiv de forma precisa y contextualizada. El sistema realiza una búsqueda semántica de documentos relevantes en un corpus optimizado de 10,000 resúmenes (abstracts) y genera respuestas coherentes y fundamentadas utilizando el modelo **Gemini 3.5 Flash**.

### ✨ Características Principales

* 🔍 **Búsqueda Semántica:** Recuperación de información eficiente mediante embeddings vectoriales y búsqueda por similitud con **FAISS**.
* 🤖 **Generación con LLM:** Generación de respuestas naturales y contextualizadas usando la API de **Gemini 3.5 Flash**.
* 💻 **Interfaz de Usuario:** Interfaz gráfica web interactiva tipo chat desarrollada con **Streamlit**.
* 📌 **Trazabilidad:** Visualización clara de las evidencias y fragmentos de texto específicos utilizados para fundamentar la respuesta.
* ⚠️ **Detección de Vacío de Información:** Capacidad para reconocer con honestidad cuando el corpus no contiene información suficiente para responder a la consulta.

---