---
sidebar_position: 1
---

# Setup Instructions

This guide will help you set up the **AgriChat-Annam** RAG chatbot on your local machine.

---

## Prerequisites

- **Python 3.8+**
- **Ollama** installed ([get it here](https://ollama.ai))
- **Git** (for cloning the repository)
- **8GB RAM** recommended

---

## Step 1: Clone Repository

```bash
git clone https://github.com/continuousactivelearning/agrichat-annam.git
cd agrichat-annam
```

---


## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 3: Download AI Models

```bash
ollama pull gemma:1b
ollama pull nomic-embed-text
```

---

## Step 4: Set Up Knowledge Base

1. Place your agricultural Q&A CSV file in `data/data.csv`
2. Ensure the CSV has columns: `questions,answers`
3. Build the vector database:

```bash
python creating_database.py
```

---

## Step 5: Start Application

In a **separate terminal**, start the Ollama server:

```bash
ollama serve
```

In your **main terminal**, start the FastAPI app:

```bash
uvicorn app:app --reload --port 8000
```

---

## Step 6: Access Web Interface

Open your browser and visit:  
👉 [http://localhost:8000](http://localhost:8000)

---


## Verification

After installation:

- You should see **"Application startup complete"** in the terminal.
- The web interface should display an input field and a submit button.
- Try asking: *"How to treat leaf curl in tomatoes?"*

---

## Troubleshooting

### Q: Models not loading?

- Verify Ollama is running:  
  ```bash
  ollama list
  ```
- Check model downloads:  
  ```bash
  ollama pull gemma:1b
  ```

### Q: ChromaDB path errors?

- Use absolute paths in Windows:  
  Example: `C:\\path\\to\\chroma_db`
- Ensure you have write permissions.

### Q: Low memory errors?

- Use a smaller model:  
  ```bash
  ollama pull gemma:1b-instruct
  ```
- Reduce ChromaDB size.

---

