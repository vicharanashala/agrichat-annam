# agrichat-annam
🌱 Agricultural RAG Chatbot
A Retrieval-Augmented Generation (RAG) chatbot for answering agricultural queries using local LLMs (Ollama), ChromaDB for vector search, and a FastAPI web interface.

Table of Contents
Features

Project Structure

Setup Instructions

1. Install Ollama

2. Download LLM and Embedding Models

3. Set Up Python Environment

4. Install Python Dependencies

5. Prepare Data

6. Build the Chroma Vector Database

7. Start the Web Application

Code Explanation

Creating the Vector Database

Query Handling Logic

Web Interface (FastAPI)

Usage

Troubleshooting

Features
Local LLMs: Uses Ollama to run models like Gemma, Llama, or Mistral locally.

Vector Search: Stores and retrieves agricultural Q&A using ChromaDB and embeddings.

Multilingual & Markdown Support: Handles Indian languages and presents answers in a readable format.

Web Interface: User-friendly chat interface built with FastAPI and Jinja2 templates.

Privacy-Preserving: All data and models run locally.

Project Structure
text
agri-chatbot/
├── data/
│   └── chilli_rows.csv         # CSV with agricultural Q&A
├── chroma_db/                  # Chroma vector store directory
├── creating_database.py        # Script to build ChromaDB
├── main.py                     # Query handling logic
├── app.py                      # FastAPI web server
├── templates/
│   └── index.html              # Web chat interface
└── README.md                   # Project documentation
Setup Instructions
1. Install Ollama
Linux:

bash
curl -fsSL https://ollama.com/install.sh | sh
Windows/Mac:
Download and run the installer from https://ollama.com/download

Verify Installation:

bash
ollama --version
2. Download LLM and Embedding Models
bash
ollama pull gemma:2b           # Small, fast LLM
ollama pull nomic-embed-text   # Embedding model for vector search
(You may choose other models as needed, e.g., llama3:8b, mistral:7b, etc.)

3. Set Up Python Environment
bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
4. Install Python Dependencies
bash
pip install langchain chromadb pandas fastapi uvicorn python-multipart markdown
5. Prepare Data
Place your Q&A CSV (e.g., chilli_rows.csv) in the data/ directory.

The CSV should have columns: questions and answers.

6. Build the Chroma Vector Database
Run the script to convert your CSV to a vector database:

bash
python creating_database.py
This will:

Read your CSV file,

Generate embeddings for each Q&A pair,

Store them in a persistent ChromaDB directory.

7. Start the Web Application
bash
uvicorn app:app --reload --port 8000
Visit http://localhost:8000 in your browser.

Code Explanation
Creating the Vector Database (creating_database.py)
Purpose: Convert CSV Q&A data into vector embeddings and store them in ChromaDB.

Key Classes:

OllamaEmbeddings: Wraps Ollama's embedding API for text.

ChromaDBBuilder: Loads CSV, creates LangChain Document objects, and stores them in ChromaDB using the embedding function.

Workflow:

Read each row from CSV.

Format as a document:
Question: ...\nAnswer: ...

Generate embeddings using Ollama's nomic-embed-text model.

Store in ChromaDB for fast retrieval.

Query Handling Logic (main.py)
Purpose: Retrieve relevant documents for a user question and generate a precise answer using an LLM.

Key Classes:

ChromaQueryHandler:

Searches ChromaDB for relevant documents using vector similarity.

Reranks results for relevance.

Constructs a prompt with context and user question.

Calls the LLM (via Ollama) to generate a concise, context-grounded answer.

Workflow:

User submits a question.

Find top-matching documents from ChromaDB.

Build a prompt with the context and question.

Send prompt to LLM (e.g., Gemma, Llama) via Ollama API.

Return the model's answer.

Web Interface (FastAPI) (app.py)
Purpose: Provide a web-based chat interface for users.

Key Components:

FastAPI server for HTTP endpoints.

Jinja2 templates for rendering HTML.

/ endpoint serves the homepage.

/query endpoint processes user questions and displays answers.

Workflow:

User visits the homepage and enters a question.

The question is sent to the /query endpoint.

The backend retrieves an answer using ChromaQueryHandler.

The answer is rendered in the chat interface.

Usage
Start Ollama server (if not running):

bash
ollama serve
Start FastAPI app:

bash
uvicorn app:app --reload --port 8000
Open http://localhost:8000 and interact with the chatbot.

Troubleshooting
Ollama not running:
Run ollama serve in a separate terminal.

Model not found:
Ensure you have pulled the correct model (ollama pull ...).

ChromaDB path errors:
Use absolute paths if needed, especially on Windows.

Memory errors:
Use smaller models (gemma:2b, tinyllama) if RAM is limited.

Contributing
Fork the repo, make changes, and submit a pull request.

For new datasets, place your CSV in data/ and update the database using the build script.

License
