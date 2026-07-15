# Sistema de Recuperación Aumentada por Generación (RAG) sobre Abstracts de arXiv

## Información General

| Campo | Información |
|-------|-------------|
| **Proyecto** | Sistema de Recuperación Aumentada por Generación (RAG) sobre Abstracts de arXiv |
| **Asignatura** | Recuperación de Información |
| **Tipo** | Examen Final |
| **Estudiante** | Kevin Xavier Alvear Cachipuendo |
| **Profesor** | Dr. Iván Carrera |
| **Fecha** | 15 de julio de 2026 |

---

## Descripción

Este proyecto implementa un sistema de **Recuperación Aumentada por Generación (Retrieval-Augmented Generation, RAG)** para responder consultas en lenguaje natural sobre un corpus de resúmenes de artículos científicos de **arXiv**.

La solución combina técnicas modernas de recuperación semántica mediante **embeddings vectoriales**, búsqueda eficiente con **FAISS**, **re-ranking** mediante **Cross-Encoder** y generación de respuestas utilizando un **Large Language Model (LLM)**.

La aplicación se presenta como una interfaz web desarrollada con **Streamlit**, permitiendo realizar consultas conversacionales y visualizar tanto la respuesta generada como las evidencias utilizadas para construirla.

---

## Características

- Recuperación semántica utilizando embeddings generados con `all-MiniLM-L6-v2`.
- Indexación vectorial mediante **FAISS (IndexFlatL2)**.
- Re-ranking opcional utilizando `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Generación de respuestas mediante **Google Gemini (`gemini-3.1-flash-lite`)**.
- Interfaz conversacional desarrollada con **Streamlit**.
- Visualización de documentos recuperados y métricas de similitud.
- Evidencias trazables para cada respuesta generada.
- Detección automática cuando el corpus no contiene información suficiente para responder una consulta.

---

## Arquitectura del Sistema

El flujo de procesamiento se compone de cinco etapas principales:

### 1. Preparación del corpus

- Carga del conjunto de datos `arxiv_data.csv`.
- Limpieza y normalización del texto.
- Tokenización.
- Eliminación de stopwords.
- Lematización.
- Combinación del título y el resumen para enriquecer la representación semántica.

### 2. Representación vectorial

- Generación de embeddings de dimensión **384** utilizando `all-MiniLM-L6-v2`.
- Almacenamiento de embeddings en formato `.npy`.
- Almacenamiento del corpus procesado en formato `.csv`.
- Construcción del índice FAISS para recuperación eficiente.

### 3. Recuperación y re-ranking

- Conversión de la consulta del usuario en un embedding.
- Recuperación inicial mediante FAISS.
- Reordenamiento opcional utilizando un Cross-Encoder para mejorar la relevancia de los resultados.

### 4. Generación de respuestas

- Construcción del contexto utilizando los documentos recuperados.
- Envío del contexto al modelo Gemini.
- Generación de respuestas restringidas exclusivamente a la información proporcionada por el corpus.

### 5. Interfaz de usuario

La aplicación ofrece:

- Chat interactivo.
- Barra lateral de configuración.
- Selección del número de documentos recuperados.
- Activación o desactivación del re-ranking.
- Visualización de documentos recuperados.
- Tabla de métricas de similitud.

---

## Tecnologías Utilizadas

| Componente | Tecnología |
|------------|------------|
| Lenguaje | Python 3.13 |
| Procesamiento de datos | Pandas, NumPy |
| Procesamiento de lenguaje natural | NLTK |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Base vectorial | FAISS (`IndexFlatL2`) |
| Re-ranking | Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) |
| Modelo generativo | Google Gemini (`gemini-3.1-flash-lite`) |
| Interfaz web | Streamlit |
| Despliegue | Streamlit Community Cloud |

---

## Estructura del Proyecto

```text
.
├── app.py
├── requirements.txt
├── README.md
├── data/
├── models/
├── notebooks/
├── src/
├── assets/
└── ...
```

> La estructura puede variar dependiendo de la organización del repositorio.

---

## Instalación

### Requisitos

- Python 3.13 o superior
- Git
- Clave de API de Google Gemini

---

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/examen-segundo-bimestre-ri.git
cd examen-segundo-bimestre-ri
```

### 2. Crear un entorno virtual

**Linux / macOS**

```bash
python -m venv venv
source venv/bin/activate
```

**Windows**

```powershell
python -m venv venv
venv\Scripts\activate
```

### 3. Instalar las dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar la API de Google Gemini

Puede configurarse de dos formas.

#### Opción 1

Crear un archivo denominado:

```text
gapi.txt
```

en la raíz del proyecto que contenga únicamente la clave de la API.

#### Opción 2

Si se utiliza **Streamlit Community Cloud**, configurar la clave desde **Secrets**.

---

### 5. Ejecutar la aplicación

```bash
streamlit run app.py
```

La aplicación estará disponible en:

```
http://localhost:8501
```

---

## Despliegue

La aplicación se encuentra desplegada en **Streamlit Community Cloud**.

**Aplicación:**

> https://kevin-alvear-examen-rag.streamlit.app/

El despliegue se realiza automáticamente a partir del repositorio de GitHub.

---

## Evaluación

El sistema fue evaluado mediante un conjunto de consultas representativas considerando los siguientes criterios:

- Corrección factual.
- Relevancia de la respuesta.
- Fidelidad al contexto recuperado.
- Capacidad de síntesis.
- Detección de información insuficiente.

Los resultados muestran que el sistema genera respuestas coherentes y fundamentadas cuando existe evidencia suficiente en el corpus y, cuando no la hay, informa explícitamente al usuario que no dispone de información para responder la consulta.

---

## Modelo de Recuperación

El sistema implementa un pipeline RAG compuesto por:

```
Consulta del usuario
        │
        ▼
Generación de embedding
        │
        ▼
Índice FAISS
        │
        ▼
Top-K documentos
        │
        ▼
Re-ranking (opcional)
        │
        ▼
Construcción del contexto
        │
        ▼
Google Gemini
        │
        ▼
Respuesta final
```

---

## Autor

**Kevin Xavier Alvear Cachipuendo**

Escuela Politécnica Nacional

Ingeniería en Ciencias de la Computación

---

## Licencia

Este proyecto fue desarrollado con fines exclusivamente académicos como parte de la asignatura **Recuperación de Información**.