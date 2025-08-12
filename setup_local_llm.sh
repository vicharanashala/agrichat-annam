#!/bin/bash

# AgriChat-Annam Local GPU Setup Script
# This script helps configure your Ollama and local Whisper setup

echo "🌱 AgriChat-Annam Full Local GPU Setup"
echo "======================================"
echo ""

echo "📋 Prerequisites:"
echo "1. Ollama must be installed and running"
echo "2. CUDA-capable GPU (recommended for best performance)"
echo "3. Your models must be pulled in Ollama"
echo ""

echo "🔧 Setup Steps:"
echo ""
echo "1. Start Ollama (if not already running):"
echo "   ollama serve"
echo ""
echo "2. Pull required models:"
echo "   ollama pull your-model-name    # Replace with your OpenAI OSS 20B model name"
echo "   ollama pull nomic-embed-text   # For embeddings"
echo ""
echo "3. Update local_llm_interface.py with your model name:"
echo "   Edit line ~98: model_name=\"your-actual-model-name\""
echo ""
echo "4. Install Whisper dependencies (if not already done):"
echo "   pip install openai-whisper torch torchaudio"
echo ""
echo "5. Test Ollama is working:"
echo "   curl http://localhost:11434/api/generate -d '{\"model\":\"your-model-name\",\"prompt\":\"Hello\"}'"
echo ""

echo "🚀 Once everything is set up:"
echo "   docker compose up --build"
echo ""

echo "✅ Your application will now use:"
echo "   - Ollama for LLM inference (your OpenAI OSS 20B model)"
echo "   - Local Whisper for speech-to-text"
echo "   - All processing on your local GPU!"
echo ""
echo "� Performance Notes:"
echo "   - Whisper 'base' model: Good balance of speed/accuracy"
echo "   - Whisper 'small': Better accuracy, slightly slower"
echo "   - Whisper 'tiny': Fastest for real-time use"
echo "   - Edit local_whisper_interface.py line ~95 to change model size"
