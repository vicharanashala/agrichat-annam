# 🌱 AgriChat-Annam: Advanced Agricultural RAG Chatbot

A sophisticated Retrieval-Augmented Generation (RAG) chatbot system for answering agricultural queries using local LLMs, CrewAI agents, multi-source knowledge retrieval (ChromaDB + Package of Practices), and an enhanced web interface with multimedia support.

---

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Architecture Overview](#architecture-overview)
- [Setup Instructions](#setup-instructions)
  - [1. Quick Start with Docker](#1-quick-start-with-docker)
  - [2. Manual Setup](#2-manual-setup)
- [API Documentation](#api-documentation)
- [Knowledge Sources](#knowledge-sources)
- [Advanced Features](#advanced-features)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## Features

### Core Capabilities
- **Multi-Modal RAG Pipeline**: Combines traditional RAG with Package of Practices (PoPs) knowledge
- **Local LLM Integration**: Uses local models for privacy-preserving inference
- **CrewAI Agents**: Intelligent agent-based query processing with specialized roles
- **Smart Recommendations**: TF-IDF + embedding-based similarity for contextual suggestions
- **Session Management**: MongoDB-backed conversation history and user sessions
- **Audio Transcription**: Whisper-based voice-to-text support

### Enhanced User Experience
- **YouTube Integration**: Displays agricultural video tutorials with thumbnails
- **Research Data Panel**: Shows sources, reasoning steps, and supporting documents
- **Real-time Responses**: Streaming API for immediate feedback
- **Multi-language Support**: Built-in language switching capabilities
- **Responsive Design**: Vercel-deployed frontend with modern UI/UX

### Data Intelligence
- **Fallback Logging**: Automatic CSV logging when LLM fallback is triggered
- **Quality Scoring**: Content quality assessment for response validation
- **Parallel Processing**: Async recommendation generation alongside main responses
- **Caching Layer**: In-memory embedding cache for performance optimization

---

## Project Structure

```bash
agrichat-annam/
├── README.md
├── docker-compose.yml              # Multi-service orchestration (MongoDB + Backend)
├── fallback_queries.csv           # Fallback LLM response logging
├── requirement.txt                # Root dependencies
├── setup_local_llm.sh             # Local LLM setup script
├── install.sh                     # Installation automation
│
├── Package_of_practices/          # Agricultural knowledge base
│   ├── Cereals_and_millets/       # Paddy, Millets, Sorghum
│   ├── Commercial_crops/
│   ├── Flowers/
│   ├── Fruits/                    # Apple, Guava, Litchi, Mango
│   ├── Medicinal_and_aromatic_plants/
│   ├── Oilseeds/                  # Niger, etc.
│   ├── Plantation_crops/          # Coconut, etc.
│   ├── Pulses/
│   ├── Spices/
│   └── Vegetables/
│
├── agrichat-backend/              # FastAPI Backend Service
│   ├── app.py                     # Main API server (session management, endpoints)
│   ├── backendRequirements.txt    # Python dependencies
│   ├── Dockerfile                 # Backend containerization
│   ├── local_whisper_interface.py # Audio transcription
│   ├── users.csv                  # User data storage
│   ├── logs/                      # Application logs
│   ├── ssl/                       # SSL certificates
│   ├── chromaDb/                  # Vector database storage
│   │   ├── [collection-id]/       # RAG collection
│   │   └── [collection-id]/       # PoPs collection
│   └── Agentic_RAG/               # Core RAG & Agent Logic
│       ├── main.py                # Main CrewAI orchestration
│       ├── chroma_query_handler.py # RAG/PoPs/LLM decision engine
│       ├── chroma_db_builder.py   # RAG database builder
│       ├── chroma_pops_builder.py # PoPs database builder
│       ├── crew_agents.py         # CrewAI agent definitions
│       ├── crew_tasks.py          # CrewAI task definitions
│       ├── local_llm_interface.py # Local LLM API wrapper
│       ├── context_manager.py     # Conversation context handling
│       ├── fast_response_handler.py # Quick response generation
│       ├── tools.py               # RAG search tools
│       ├── database_config.py     # Database configuration
│       ├── Db - Sheet1.csv        # Agricultural Q&A dataset
│       └── fallback_queries.csv   # Local fallback logging
│
├── agrichat-frontend/             # Modern Web Interface
│   ├── index.html                 # Main chat interface
│   ├── script.js                  # Frontend logic (YouTube, research panel, chat)
│   ├── style.css                  # Responsive styling
│   ├── config.js                  # Frontend configuration
│   ├── ssl-override.js            # SSL handling
│   ├── Dockerfile                 # Frontend containerization
│   ├── vercel.json               # Vercel deployment config
│   └── .vercel/                  # Vercel deployment artifacts
│
└── docs/                         # Docusaurus Documentation Site
    ├── docusaurus.config.js      # Docs configuration
    ├── package.json              # Node.js dependencies
    ├── sidebars.js               # Documentation navigation
    ├── blog/                     # Blog posts
    ├── docs/                     # Documentation pages
    │   ├── getting-started/      # Setup guides
    │   ├── Code-Documentation/   # Technical documentation
    │   ├── Dataset/              # Data documentation
    │   ├── contributing/         # Contribution guides
    │   └── troubleshooting/      # Common issues
    ├── src/                      # Docusaurus source
    └── static/                   # Static assets
````

---

## Architecture Overview

### RAG Pipeline Flow
```
User Query → FastAPI Backend → Decision Engine
                              ↓
                          ChromaDB Search (RAG)
                              ↓
                          PoPs Search (Fallback)
                              ↓
                          LLM Generation (Final Fallback)
                              ↓
                          Response + Research Data + YouTube Links
```

### Multi-Source Knowledge Integration
1. **Primary RAG**: ChromaDB vector search on agricultural Q&A dataset
2. **Package of Practices (PoPs)**: Structured agricultural knowledge base with quality scoring
3. **LLM Fallback**: Local language model for queries not covered by structured data
4. **Recommendation Engine**: TF-IDF + embeddings for contextual question suggestions

---

## Setup Instructions

### 1. Quick Start with Docker

**Prerequisites:**
- Docker and Docker Compose installed
- Minimum 8GB RAM (for local LLM models)

**Launch the full stack:**
```bash
# Clone repository
git clone <repository-url>
cd agrichat-annam

# Start all services (MongoDB + Backend)
docker-compose up -d

# Build knowledge databases (first time only)
docker-compose exec backend python Agentic_RAG/chroma_db_builder.py
docker-compose exec backend python Agentic_RAG/chroma_pops_builder.py
```

**Access Points:**
- Backend API: `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`
- Frontend: Deploy to Vercel or serve statically

### 2. Manual Setup

**Step 1: Local LLM Setup**
```bash
# Run the automated setup script
./setup_local_llm.sh

# Or manually install Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b        # Primary LLM model
ollama pull gemma3:27b         # Fallback model
ollama pull nomic-embed-text # Embedding model
ollama serve                # Start Ollama server
```

**Step 2: Database Setup**
```bash
# Start MongoDB
docker run -d -p 27017:27017 --name mongodb mongo:latest

# Or install MongoDB locally
# Follow MongoDB installation guide for your OS
```

**Step 3: Backend Setup**
```bash
cd agrichat-backend

# Install dependencies
pip install -r backendRequirements.txt

# Build knowledge databases
python Agentic_RAG/chroma_db_builder.py    # Build RAG database
python Agentic_RAG/chroma_pops_builder.py  # Build PoPs database

# Start backend server
uvicorn app:app --reload --port 8000
```

**Step 4: Frontend Deployment**
```bash
cd agrichat-frontend

# Option 1: Deploy to Vercel (recommended)
npm install -g vercel
vercel --prod

# Option 2: Serve locally
python -m http.server 3000
```

---

## API Documentation

### Core Endpoints

| Method | Endpoint | Description | Key Features |
|--------|----------|-------------|--------------|
| `POST` | `/api/query` | Process user questions | RAG/PoPs/LLM pipeline, research data |
| `POST` | `/api/query/stream` | Streaming responses | Real-time answer generation |
| `POST` | `/api/recommendations` | Get question suggestions | TF-IDF + embedding similarity |
| `GET` | `/api/sessions` | List user sessions | MongoDB session management |
| `POST` | `/api/session/{id}/query` | Session-specific queries | Conversation context |
| `POST` | `/api/transcribe-audio` | Voice-to-text | Whisper integration |

### Authentication & Session Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/auth/login` | User authentication |
| `GET` | `/api/session/{session_id}` | Get session history |
| `POST` | `/api/toggle-status/{session_id}/{status}` | Update session status |
| `DELETE` | `/api/delete-session/{session_id}` | Delete session |
| `GET` | `/api/export/csv/{session_id}` | Export session data |

### Utility Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Service health check |
| `POST` | `/api/update-language` | Change response language |
| `POST` | `/api/session/{session_id}/rate` | Rate responses |

### Response Format

**Standard Query Response:**
```json
{
  "answer": "Response text with markdown formatting",
  "reasoning_steps": ["Step 1", "Step 2", "Step 3"],
  "research_data": [
    {
      "title": "Document title",
      "content": "Relevant excerpt",
      "source": "PoPs/RAG/Fallback LLM",
      "similarity_score": 0.85,
      "youtube_url": "https://youtube.com/watch?v=..."
    }
  ],
  "source": "RAG/PoPs/Fallback LLM",
  "session_id": "uuid-string"
}
```

---

## Knowledge Sources

### 1. Package of Practices (PoPs)
- **Content**: 2000+ structured agricultural documents
- **Coverage**: Cereals, fruits, vegetables, commercial crops, spices
- **Format**: JSON documents with metadata and YouTube links
- **Quality Control**: Automated relevance scoring and LLM verification

### 2. RAG Database
- **Source**: `Agentic_RAG/Db - Sheet1.csv`
- **Content**: Curated agricultural Q&A pairs
- **Processing**: Vector embeddings with ChromaDB storage
- **Retrieval**: Cosine similarity search with configurable thresholds

### 3. Fallback LLM
- **Models**: Gemma:1b, Llama variants via Ollama
- **Usage**: When structured knowledge insufficient
- **Logging**: Automatic CSV logging for improvement tracking
- **Integration**: CrewAI agents for specialized responses

---

## Advanced Features

### YouTube Video Integration
- Automatic extraction of YouTube URLs from knowledge sources
- Thumbnail generation with video titles
- Embedded video player in research panel
- "Suggested Video" labels for enhanced UX

### Quality Assurance
- **Content Quality Scoring**: Specificity, relevance, completeness metrics
- **PoPs Verification**: LLM-based relevance checking before acceptance
- **Strong Match Bypass**: High-confidence matches with validation
- **Fallback Tracking**: CSV logging when structured knowledge fails

### Performance Optimization
- **Embedding Cache**: In-memory caching for repeated queries
- **Parallel Processing**: Async recommendations alongside main responses
- **TF-IDF Prefiltering**: Efficient candidate selection before embedding comparison
- **Response Deduplication**: Canonical similarity to prevent duplicate suggestions

### Development Tools
- **Health Monitoring**: `/health` endpoint for service status
- **Database Testing**: `/api/test-database-toggle` for switching knowledge sources
- **Session Export**: CSV export for conversation analysis
- **Streaming Responses**: Real-time answer generation with `/api/query/stream`

---

## Usage

### Basic Query Flow
1. **Send Question**: POST to `/api/query` with user question
2. **Knowledge Search**: System searches RAG → PoPs → LLM in order
3. **Response Generation**: Returns answer with reasoning and sources
4. **Research Panel**: Frontend displays supporting documents and videos

### Session Management
1. **Login**: Authenticate user via `/api/auth/login`
2. **Create Session**: Start new conversation thread
3. **Contextual Queries**: Use session ID for conversation continuity
4. **Export/Delete**: Manage session data as needed

### Recommendation System
1. **Auto-Suggestions**: Parallel generation of 4 contextual questions
2. **Similarity Ranking**: TF-IDF + embedding-based scoring
3. **RAG Preference**: Prioritizes questions with knowledge base coverage
4. **Deduplication**: Prevents repetitive suggestions

---

## Troubleshooting

### Common Issues

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **Ollama Not Running** | "Connection refused" errors | Run `ollama serve` in separate terminal |
| **Models Not Found** | "Model not available" errors | Run `ollama pull gemma:1b` and `ollama pull nomic-embed-text` |
| **MongoDB Connection** | Database connection errors | Ensure MongoDB running on port 27017 |
| **ChromaDB Path Issues** | Vector search failures | Check ChromaDB permissions in `agrichat-backend/chromaDb/` |
| **Low Memory Errors** | OOM or slow responses | Use smaller models (gemma:1b vs gemma:7b) |
| **Docker Volume Issues** | CSV logging not visible | Mount volumes for fallback_queries.csv |
| **Frontend API Errors** | CORS or connection issues | Verify backend URL in `config.js` |

### Performance Tuning

**For High-Memory Systems (16GB+):**
```bash
ollama pull gemma:7b      # Use larger, more capable model
ollama pull llama2:13b    # Alternative high-performance model
```

**For Low-Memory Systems (8GB):**
```bash
ollama pull gemma:1b      # Lightweight model
ollama pull tinyllama     # Ultra-lightweight alternative
```

**Database Optimization:**
```bash
# Rebuild indexes for faster retrieval
python Agentic_RAG/chroma_db_builder.py --rebuild-index
python Agentic_RAG/chroma_pops_builder.py --rebuild-index
```

### Development Mode

**Enable Debug Logging:**
```bash
export LOG_LEVEL=DEBUG
export USE_FAST_MODE=false          # Disable fast responses for debugging
export DISABLE_RECOMMENDATIONS=true  # Disable recommendations for testing
```

**Database Switching:**
```bash
# Test with different knowledge sources
curl -X POST http://localhost:8000/api/test-database-toggle \
  -H "Content-Type: application/json" \
  -d '{"use_rag": true, "use_pops": false}'
```

---

## Contributing

### Development Workflow
1. **Fork Repository**: Create your feature branch from `main`
2. **Local Setup**: Follow manual setup instructions
3. **Database Testing**: Rebuild knowledge bases after data changes
4. **API Testing**: Use `/docs` endpoint for interactive API testing
5. **Frontend Testing**: Deploy to Vercel preview for UI changes
6. **Pull Request**: Submit with comprehensive description

### Code Standards
- **Backend**: Follow FastAPI async patterns, use type hints
- **Frontend**: Modern JavaScript (ES6+), responsive CSS
- **Documentation**: Update README for new features
- **Testing**: Test API endpoints with sample queries

### Key Areas for Contribution
- **Knowledge Sources**: Add new agricultural datasets
- **Model Integration**: Support for additional LLM providers
- **UI/UX**: Enhanced chat interface and visualization
- **Performance**: Optimization for large-scale deployment
- **Multilingual**: Expand language support beyond English

---

**Live Demo**: [Frontend URL] | **API Docs**: [Backend URL]/docs | **Documentation**: [Docs URL]