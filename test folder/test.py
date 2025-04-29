# import streamlit as st
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain.chains import ConversationalRetrievalChain
# from langchain_ollama import OllamaLLM, OllamaEmbeddings
# import os



# @st.cache_resource
# def init_models():
#     os.environ["OLLAMA_HOST"] = "http://localhost:11434/v1"  # Set host via env var
#     return OllamaLLM(
#         model="deepseek-r1:1.5b",
#         temperature=0.1,
#         num_ctx=2048,
#         num_thread=4
#     ), OllamaEmbeddings(
#         model="nomic-embed-text:latest"
#     )

# ollama, embeddings = init_models()

# @st.cache_resource
# def load_vector_store(_embeddings):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=500,
#         chunk_overlap=50,
#         length_function=len
#     )
#     if not os.path.exists('vector_store'):
#         documents = []
#         data_dir = 'datasets'
#         if not os.path.exists(data_dir):
#             os.makedirs(data_dir)
#             st.error("Data directory created. Please add PDFs to the 'data' folder and restart the app.")
#             st.stop()
#         for pdf_file in os.listdir(data_dir):
#             if pdf_file.endswith('.pdf'):
#                 pdf_path = os.path.join(data_dir, pdf_file)
#                 loader = PyPDFLoader(pdf_path)
#                 pages = loader.load()
#                 documents.extend(pages)
#         if not documents:
#             st.error("No PDFs found in 'data' directory. Please add PDFs and restart the app.")
#             st.stop()
#         splits = text_splitter.split_documents(documents)
#         vector_store = FAISS.from_documents(splits, _embeddings)
#         vector_store.save_local('vector_store')
#     else:
#         vector_store = FAISS.load_local('vector_store', _embeddings, allow_dangerous_deserialization=True)
#     return vector_store

# vector_store = load_vector_store(embeddings)

# @st.cache_resource
# def create_qa_chain(_vector_store, _llm):  # Add underscores to both parameters
#     return ConversationalRetrievalChain.from_llm(
#         llm=_llm,  # Use the underscore-prefixed parameter
#         retriever=_vector_store.as_retriever(search_kwargs={"k": 3}),
#         return_source_documents=True,
#         verbose=False
#     )

# # The rest of your code remains the same
# qa_chain = create_qa_chain(vector_store, ollama)

# st.title("PDF Chat with RAG")

# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# user_question = st.text_input("Ask a question about your PDFs:")

# if user_question:
#     try:
#         response = qa_chain.invoke({
#             "question": user_question,
#             "chat_history": st.session_state.chat_history
#         })
#     except Exception as e:
#         st.error(f"Error: {str(e)}")
    
#     st.session_state.chat_history.append((user_question, response["answer"]))
    
#     if len(st.session_state.chat_history) > 10:
#         st.session_state.chat_history = st.session_state.chat_history[-10:]
    
#     for question, answer in st.session_state.chat_history:
#         st.write(f"Q: {question}")
#         st.write(f"A: {answer}")
#         st.write("---")

# if st.button("Clear Chat"):
#     st.session_state.chat_history = []


import re

def extract_think_content(text):
    # Use regex to find content between <think> and </think> tags
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if match:
        return match.group(1).strip()  # Extract and remove leading/trailing whitespace
    return None  # Return None if no match is found

# Example input text
input_text = """
Answer: <think>
Okay, I need to figure out what mobile veterinary units are based on the given context. Let me read through it again.

The context starts by mentioning that mobile veterinary units function in the department aiming for door-to-door health care and veterinary services for livestock in remote areas. There are 56 of these units functioning in the state now.

Then, there's information about acquiring 31 vehicles under the national agriculture development program from 2012-13 to hand over to the department officials. These units have a head of a Veterinary Assistant, which probably means they're led by trained veterinary assistants.

From 2014-15, the units provided service to about 3.98 lakh livestock and did 0.92 lakh artificial inseminations for poultry.

The main objectives are delivering health coverage and veterinary assistance to farmers through their doorsteps.

I need to write a comprehensive answer addressing what mobile veterinary units are, how many there are, the vehicles they use, the subjects they service, and their aims.

First, I'll start by stating that mobile veterinary units are specialized teams assigned in remote areas to provide welfare care. They have a clear aim to ensure animals are healthy and meet all dietary requirements.

Next, I should mention the number of units: 56 with two main purposes—conservation and agricultural welfare. However, more importantly, 
they function in the state with the help of some vehicles.

Then, there's info about acquiring additional vehicles during 2012-13 for Mobile Veterinary Units to improve service. This would be important enough that I include it in my answer.

Finally, the context mentions services provided to livestock and artificial inseminations. It doesn't mention human inseminations much but gives animal information. The aim is clearly stated again, so I'll make sure to highlight that.

I should also note how these units are recognized by the public because they deliver directly on farmers' doors without long-distance trips or private vehicles.
</think>

Mobile veterinary units ( MVUs) are specialized teams assigned in remote areas in the state to provide welfare care to animals. They aim for health coverage and maternal and fetal-coverage for livestock. MVUs consist of two main purposes: conservation and agricultural welfare. Currently, 56 MVUs operate under the Department aiming to deliver welfare services door-to-door.

These units utilize vehicles acquired during 2012-13 from the National Agriculture Development Programme, providing Mobile Veterinary Units with improved service. Their goals include offering health care and veterinary assistance directly on the farmers' doors in rural counties.

MVUs also conduct artificial inseminations for poultry. They are recognized by the public as efficient alternatives to private vehicle trips, with 56 of them functioning efficiently with some additional vehicles.
"""

# Extract content within <think> tags
think_content = extract_think_content(input_text)

# Output result
if think_content:
    print("Content within <think> tags:")
    print(think_content)
else:
    print("No content found within <think> tags.")
