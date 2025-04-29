---
sidebar_position: 2
---
# Project Structure

📁 **Project Structure**  
This project is organized in a simple and modular format to separate the dataset, vector database, core logic, and web interface. Below is an overview of the core structure:

---

```bash
agrichat-annam/
├── data/                      # Agricultural Q&A dataset
│   └── data.csv
├── chroma_db/                  # Vector storage directory for embeddings
├── creating_database.py        # Script to build ChromaDB from CSV
├── main.py                     # Core query logic for RAG pipeline
├── app.py                      # FastAPI server application
├── requirements.txt            # Dependency list for Python environment
└── templates/                  # Frontend templates and static assets
    ├── index.html              # Web chat interface (HTML)
    └── static/                 # Static files (CSS, JS)
        └── style.css           # Styling for the frontend
```

---

🔙 **Key Components**

- **data/**:  
  Contains the CSV file (`data.csv`) used to generate the knowledge base for retrieval.

- **chroma_db/**:  
  Stores the generated Chroma vector database built from the CSV file. This enables fast semantic search.

- **creating_database.py**:  
  Script that reads the CSV data, generates embeddings using the `nomic-embed-text` model, and saves the vectors into ChromaDB.

- **main.py**:  
  Core logic for processing user queries, fetching relevant answers from the database, and interacting with the AI model.

- **app.py**:  
  FastAPI server that provides API endpoints for frontend interaction. It processes user questions and returns AI-generated answers.

- **templates/**:  
  Contains all frontend resources:
  - `index.html`: Main user interface for the chat application.
  - `static/style.css`: Styling for the web interface to enhance user experience.

---

🔍 **Notes**

- **Simple RAG Pipeline**:  
  The system uses a lightweight retrieval-augmented generation (RAG) approach — retrieving relevant data before answering.

- **Frontend Static Files**:  
  HTML templates are supported by static CSS files to make the interface clean and usable.

- **Local-First Deployment**:  
  Designed for quick local deployment using FastAPI with minimal dependencies.

---

