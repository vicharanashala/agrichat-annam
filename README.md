# agrichat-annam
# 🌱 AgriChat-Annam: Agricultural RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for answering agricultural queries using local LLMs (Ollama), ChromaDB for vector search, and a FastAPI web interface.

---

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
  - [1. Install Ollama](#1-install-ollama)
  - [2. Download LLM and Embedding Models](#2-download-llm-and-embedding-models)
  - [3. Install Python Dependencies](#3-install-python-dependencies)
  - [4. Prepare Data](#4-prepare-data)
  - [5. Build the Chroma Vector Database](#5-build-the-chroma-vector-database)
  - [6. Start the Web Application](#6-start-the-web-application)
- [Code Explanation](#code-explanation)
  - [Creating the Vector Database](#creating-the-vector-database)
  - [Query Handling Logic](#query-handling-logic)
  - [Web Interface (FastAPI)](#web-interface-fastapi)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## Features
- **Local LLMs**: Uses Ollama to run models like Gemma, Llama locally
- **Vector Search**: Stores and retrieves agricultural Q&A using ChromaDB and embeddings
- **Web Interface**: User-friendly chat interface built with FastAPI and Jinja2 templates
- **Privacy-Preserving**: All data and models run locally

---

## Project Structure
agri-chatbot/
├── data/
│ └── data.csv # CSV with agricultural Q&A
├── chroma_db/ # Chroma vector store directory
├── creating_database.py # Script to build ChromaDB
├── main.py # Query handling logic
├── app.py # FastAPI web server
├── templates/
│ └── index.html # Web chat interface
└── README.md # Project documentation

text

---

## Setup Instructions

### 1. Install Ollama
**Linux:**
curl -fsSL https://ollama.com/install.sh | sh

text

**Windows/Mac:**  
Download installer from [ollama.ai](https://ollama.ai)

**Verify Installation:**
ollama --version

text

---

### 2. Download LLM and Embedding Models
ollama pull gemma:1b
ollama pull nomic-embed-text

text

---

### 3. Install Python Dependencies
pip install langchain chromadb pandas fastapi uvicorn python-multipart markdown

text

---

### 4. Prepare Data
1. Place your Q&A CSV in the `data/` directory
2. Ensure CSV has columns: `questions` and `answers`

---

### 5. Build the Chroma Vector Database
python creating_database.py

text

---

### 6. Start the Web Application
uvicorn app:app --reload --port 8000

text
Visit [http://localhost:8000](http://localhost:8000)

---

## Code Explanation

### Creating the Vector Database (`creating_database.py`)
- Converts CSV Q&A data into vector embeddings
- Uses `OllamaEmbeddings` wrapper for text embeddings
- Stores documents in ChromaDB using `ChromaDBBuilder`

### Query Handling Logic (`main.py`)
- `ChromaQueryHandler` class handles:
  - Vector similarity search
  - Result reranking
  - LLM prompt construction
  - Response generation via Ollama API

### Web Interface (FastAPI) (`app.py`)
- FastAPI server with two endpoints:
  - `GET /`: Renders chat interface
  - `POST /query`: Processes user questions
- Converts markdown responses to HTML

---

## Usage
1. Start Ollama server:
ollama serve

text

2. Launch application:
uvicorn app:app --reload --port 8000

text

3. Access chat interface at `http://localhost:8000`

---

## Troubleshooting
| Issue | Solution |
|-------|----------|
| Ollama not running | Run `ollama serve` in separate terminal |
| Model not found | Verify with `ollama list` and pull required models |
| ChromaDB path errors | Use absolute paths for Windows systems |
| Low memory errors | Use smaller models like `gemma:1b` |

---

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

**License:** MIT  
**Maintainer:** [Your Name]  
**Documentation:** [Add Documentation Link if Available]
