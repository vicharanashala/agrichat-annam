import pandas as pd
import os
import json
import networkx as nx
from llm_clients.wrapper import LLMWrapper



def llm_ner(text, labels, model_name='gemini-2.5-flash'):
    """
    Identifies named entities, their types, and relations from text using an LLM.

    Args:
        text (str): The input text to process.
        labels (list): A list of entity labels to guide the LLM.
        model_name (str): The name of the LLM model to use.

    Returns:
        dict: A dictionary containing the identified entities and relations in JSON format.
    """
    
    system_prompt = f"""
        You are an expert AI system designed to build knowledge graphs. Your task is to extract named entities and the relationships between them from the provided text.

        **Instructions:**

        1.  **Extract Entities:** First, identify every named entity in the text. For each entity, provide its exact text, assign it a type from the provided labels, and give it a unique sequential ID (e.g., "e1", "e2").
        2.  **Extract Relationships:** Second, identify all relationships between the entities. A relationship is a triplet of (subject, predicate, object). Represent this using the unique IDs of the entities you identified.

        **Entity Labels:**
        {', '.join(labels)}

        **Output Format:**
        You MUST provide the output in a single, strict JSON object with two keys: "entities" and "relations". Do not add any text before or after the JSON object.

        **Example Output Structure:**
        ```json
        {{
          "entities": [
            {{
              "id": "e1",
              "text": "<The exact text of the entity>",
              "type": "<The category from the entity labels>"
            }}
          ],
          "relations": [
            {{
              "subject_id": "<The ID of the subject entity>",
              "predicate": "<The verb phrase describing the relationship>",
              "object_id": "<The ID of the object entity>"
            }}
          ]
        }}


        **Input Text:**

        {text}
        """

    llm_client = LLMWrapper(client_type='gemini') 
    
    raw_response = llm_client.generate_text(prompt=system_prompt, model=model_name) 
    
    # Extract the JSON part of the response
    try:
        json_response = json.loads(raw_response.strip().replace('```json', '').replace('```', ''))
        return json_response
    except json.JSONDecodeError:
        print('Failed to decode JSON from the LLM response.')
        return None

def format_chunk_to_text(chunk_df):
    """
    Converts a pandas DataFrame chunk into a single block of natural language text.
    Each row is formatted as "Header1: Value1, Header2: Value2, ..."
    """
    headers = chunk_df.columns.tolist()
    formatted_rows = []

    for _, row in chunk_df.iterrows():
        row_parts = []
        for header in headers:
            value = row[header]
            # Ensure we don't include empty/NaN values in our string
            if pd.notna(value) and str(value).strip():
                row_parts.append(f"{header}: {value}")
        
        # Join all "header: value" pairs for a single row
        formatted_row_string = ", ".join(row_parts)
        formatted_rows.append(formatted_row_string)
        
    # Join all formatted rows with a newline character to form the final text block
    return "\n".join(formatted_rows)


def create_kg_from_json(json_data):
    """
    Creates a knowledge graph from the JSON output of the LLM NER model.

    Args:
        json_data (dict): The JSON data containing entities and relations.

    Returns:
        networkx.DiGraph: The created knowledge graph.
    """
    if not json_data or "entities" not in json_data or "relations" not in json_data:
        return None

    G = nx.DiGraph()

    # Add nodes for each entity
    for entity in json_data["entities"]:
        G.add_node(entity["id"], text=entity["text"], type=entity["type"])

    # Add edges for each relation
    for relation in json_data["relations"]:
        G.add_edge(relation["subject_id"], relation["object_id"], predicate=relation["predicate"])

    return G


def main():

    # Import and prepare labels
    from ner_labels import labels
    labels = [l.lower() for l in labels]

    data_path = "data/kcc_dataset_part_1.csv"
    chunk_size = 5  # Smaller chunk size for testing

    try:
        csv_reader = pd.read_csv(data_path, chunksize=chunk_size, iterator=True)
        print(f"Reading '{data_path}' in chunks of {chunk_size} rows...")

        for i, chunk in enumerate(csv_reader):
            print(f"\n--- Processing Chunk {i+1} ---")

            text_chunk = format_chunk_to_text(chunk)

            if not text_chunk.strip():
                print("Skipping empty chunk.")
                continue
            
            print("Running NER model on the formatted text...")
            
            ner_results = llm_ner(text_chunk, labels)

            if ner_results:
                print("--- NER Results ---")
                print(json.dumps(ner_results, indent=2))
                
                print("\n--- Creating Knowledge Graph ---")
                kg = create_kg_from_json(ner_results)
                
                if kg:
                    print(f"Number of nodes: {kg.number_of_nodes()}")
                    print(f"Number of edges: {kg.number_of_edges()}")
                    
                    print("\nNodes:")
                    for node, data in kg.nodes(data=True):
                        print(f"  - {node}: {data}")
                        
                    print("\nEdges:")
                    for u, v, data in kg.edges(data=True):
                        print(f"  - ({u}) -> ({v}): {data}")
                else:
                    print("Could not create knowledge graph from the provided JSON.")


    except FileNotFoundError:
        print(f"Error: The file '{data_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
