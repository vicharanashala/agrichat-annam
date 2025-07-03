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

1. Place your agricultural Q&A CSV file in `agrichat-backend/<current-pipeline>/data/sample_data.csv`
2. Ensure the CSV has columns: `questions,answers`
3. Build the vector database:

```bash
cd agrichat-backend/<current-pipeline>
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
uvicorn agrichat-backend.app:app --reload --port 8000
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

Perfect — thanks for clarifying!

Since your **Docker setup is based on `RAGpipelinev3/Gemini_based_processing`**, not `Ollama`, you should:

* ✅ Clearly state that the **Docker setup runs the Gemini-based pipeline**.
* ✅ Remove any mention of `ollama` in the Docker section.
* ✅ Mention that **API keys for Gemini** must be set (likely via `.env`).

---

### ✅ Updated **Docker Setup** Section (for Gemini-based deployment)

You can add this at the end of your **Docusaurus `Setup Instructions` page**:

---

## 🐳 Optional: Docker Setup (Gemini Pipeline)

You can run the chatbot using Docker with the **Gemini-based RAG pipeline**.

### Step 1: Set Environment Variables

Create a `.env` file in the project root with your **Google Gemini API key**:

```env
MONGO_URI=your-mongo-atlas-uri
GEMINI_API_KEY=your-gemini-api-key
```

---

### Step 2: Build and Run Containers

In the project root, run:

```bash
docker-compose up --build
```

This will:

* Start the **FastAPI backend** with the `Gemini_based_processing` pipeline.
* Serve the **frontend** via **NGINX**.
* Make both accessible via Docker’s internal network.

---

### Step 3: Access the App

Open your browser and go to:

👉 [http://localhost](http://localhost)

You should see the chatbot UI.

---

### Notes

* Ensure you have a valid **Gemini API key** and internet access.
* The knowledge base used is located at:

  ```
  agrichat-backend/RAGpipelinev3/Gemini_based_processing/Data/sample_data.csv
  ```
* If you update the data, re-run `creating_database.py` before rebuilding the image.

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

