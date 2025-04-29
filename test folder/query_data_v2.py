import chromadb


def retrieve_answer(query, chroma_collection):
 """Retrieves the answer from ChromaDB based on the query.


 This version prioritizes semantic similarity AND question matching,
 but does filtering in Python due to ChromaDB limitations.
 """


 # Step 1: Perform similarity search WITHOUT metadata filtering
 results = chroma_collection.query(
 query_texts=[query],
 n_results=5,  # Increase n_results to get more candidates
 # Removed where clause
 )


 if results and results['metadatas']:
    filtered_results = []
    for i, metadata in enumerate(results['metadatas']):
        if metadata and metadata.get('question') and query.lower() in metadata['question'].lower():
            filtered_results.append((i, metadata))


 if filtered_results:
    top_index, top_metadata = filtered_results[0]
    return results['metadatas'][top_index]['answer']
 else:
    return None


if __name__ == "__main__":
 client = chromadb.PersistentClient(path="chroma_db")
 collection = client.get_collection(name="faq_collection")


 query = "Describe about the control measure fungal wilt"
 answer = retrieve_answer(query, collection)


 if answer:
    print(f"Query: {query}")
    print(f"Retrieved Answer: {answer}")
 else:
    print("No matching answer found.")
