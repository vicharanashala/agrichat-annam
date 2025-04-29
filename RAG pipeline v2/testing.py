import requests

# ==== CONFIGURATION ====
OLLAMA_SERVER_URL = "http://192.168.1.67:11434"  # Replace with your actual Ollama GPU server IP
MODEL_NAME = "nomic-embed-text:latest"
BATCH_SIZE = 32  # Tune based on your GPU

# ==== INPUT TEXT ====
texts = [
    "Artificial Intelligence is transforming the world.",
    "Ollama allows local LLM inference with GPU acceleration.",
    "Embeddings are dense vector representations of text.",
    "This example demonstrates fast embedding generation."
] * 10  # Simulate a bigger batch

# ==== CHUNKING INTO BATCHES ====
def batch_chunks(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

# ==== EMBEDDING FUNCTION ====
def get_embeddings(batch):
    response = requests.post(
        f"{'http://192.168.1.67:11434/v1'}/api/embeddings",
        json={"model": MODEL_NAME, "input": batch}
    )
    response.raise_for_status()
    return response.json()["data"]

# ==== MAIN EMBEDDING PROCESS ====
all_embeddings = []
for batch in batch_chunks(texts, BATCH_SIZE):
    embeddings = get_embeddings(batch)
    all_embeddings.extend(embeddings)

# ==== RESULT ====
print(f"Generated {len(all_embeddings)} embeddings.")
print("Sample vector (first text):", all_embeddings[0][:5])  # Print first 5 values of the first embedding
