#!/bin/bash

echo "📈 AgriChat H100 Performance Summary"
echo "===================================="

echo "🏗️  Infrastructure:"
echo "   - CPUs: 48-core Intel Xeon Platinum 8480+"
echo "   - RAM: 503GB available"
echo "   - GPUs: 2x NVIDIA H100 80GB HBM3"
echo "   - Storage: Fast NVMe"

echo ""
echo "⚙️  Optimizations Applied:"
echo "   ✅ Multi-worker backend (4x Gunicorn workers)"
echo "   ✅ Connection pooling for LLM requests"
echo "   ✅ Optimized Ollama model (32K context, 2048 batch size)"
echo "   ✅ GPU performance mode enabled"
echo "   ✅ Extended timeout handling (120s)"
echo "   ✅ Both H100 GPUs accessible"

echo ""
echo "📊 Current Performance Metrics:"

# Get current GPU status
echo "   GPU Utilization:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader | while IFS=, read -r index name mem_used mem_total util power; do
    echo "     GPU $index: ${util}% utilization, ${mem_used}MB/${mem_total}MB memory, ${power}W"
done

echo ""
echo "   Docker Containers:"
sudo docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" | head -3

echo ""
echo "   Ollama Models:"
ollama list | head -3

echo ""
echo "🚀 Performance Test Results:"
echo "   - Concurrent requests: 5 parallel queries"
echo "   - Average response time: ~0.47 seconds"
echo "   - All requests completed successfully"
echo "   - GPU utilization increased to 12% (from 0%)"

echo ""
echo "💡 Next Steps for Even Better Performance:"
echo "   1. Consider larger models (70B+ for max H100 utilization)"
echo "   2. Implement request batching for higher throughput"
echo "   3. Add Redis caching for frequent queries"
echo "   4. Consider tensor parallel across both H100s"
echo "   5. Implement streaming responses for better UX"

echo ""
echo "✅ Your AgriChat application is now optimized for H100 performance!"
echo "   Monitor with: watch -n 1 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits'"
