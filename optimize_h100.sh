#!/bin/bash

echo "🚀 AgriChat H100 Performance Optimization Script"
echo "=============================================="

# 1. Create optimized model
echo "📦 Creating optimized Gemma3 model..."
ollama create gemma3-optimized -f Modelfile.optimized

# 2. Set Ollama environment variables for H100 optimization
echo "⚙️ Setting Ollama environment variables..."
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_ORIGINS="*"
export OLLAMA_HOST="0.0.0.0:11434"
export CUDA_VISIBLE_DEVICES=0,1

# 3. Restart Ollama with optimizations
echo "🔄 Restarting Ollama with optimizations..."
sudo systemctl restart ollama
sleep 5

# 4. Pre-load the optimized model
echo "🔥 Pre-loading optimized model..."
ollama run gemma3-optimized "warmup" --verbose

# 5. Set GPU performance mode
echo "⚡ Setting GPU performance mode..."
sudo nvidia-smi -pm 1
sudo nvidia-smi -ac 2619,1980  # Max memory and graphics clock for H100

# 6. Restart containers with new configuration
echo "🐳 Restarting containers..."
sudo docker compose down
sudo docker compose up -d --build

echo "✅ Optimization complete!"
echo ""
echo "📊 Performance improvements:"
echo "- Increased context size: 32K tokens"
echo "- Increased batch size: 2048"
echo "- Multi-GPU utilization enabled"
echo "- Connection pooling enabled"
echo "- Concurrent request handling: 4 workers"
echo ""
echo "🔍 Monitor performance with:"
echo "  watch -n 1 nvidia-smi"
echo "  sudo docker stats"
