from llm_clients import LLMWrapper

# Example usage:

# To use the Gemini client
# Make sure to set your GEMINI_API_KEY in the llm_clients/config.py file or as an environment variable
try:
    gemini_wrapper = LLMWrapper(client_type="gemini")
    prompt = "Write a PHD Thesis on political idiocy"
    response = gemini_wrapper.generate_text(prompt, model="gemini-2.5-flash")
    print("Gemini Response:", response)
except ValueError as e:
    print(e)


# To use the Ollama client
# Make sure Ollama is running locally
try:
    ollama_wrapper = LLMWrapper(client_type="ollama")
    prompt = "How many r's in strawberry?"
    response = ollama_wrapper.generate_text(prompt, model="llama3.1")
    print("Ollama Response:", response)
except Exception as e:
    print(f"Error with Ollama client: {e}")

