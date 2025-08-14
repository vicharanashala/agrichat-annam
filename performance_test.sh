#!/bin/bash

echo "🧪 AgriChat H100 Performance Test"
echo "================================="

# Function to run performance test
run_test() {
    echo "Test $1/5: $2"
    start_time=$(date +%s.%N)
    curl -s http://localhost:11434/api/generate -d "{
        \"model\":\"gemma3-optimized\",
        \"prompt\":\"$2\",
        \"stream\":false,
        \"options\":{\"temperature\":0.3,\"num_predict\":200}
    }" | jq -r '.response' | wc -w
    end_time=$(date +%s.%N)
    duration=$(echo "$end_time - $start_time" | bc)
    echo "Duration: ${duration}s"
    echo "---"
}

echo "Starting performance tests..."
echo "GPU Status before tests:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

echo -e "\n🚀 Running 5 concurrent performance tests..."

# Run tests in parallel
{
    run_test 1 "Explain sustainable farming practices for wheat cultivation." &
    run_test 2 "What are the best irrigation methods for rice farming?" &  
    run_test 3 "How can farmers prevent soil erosion in hilly areas?" &
    run_test 4 "Describe integrated pest management strategies." &
    run_test 5 "What are the benefits of organic fertilizers?" &
    wait
} 

echo -e "\n📊 GPU Status after tests:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

echo -e "\n✅ Performance test completed!"
