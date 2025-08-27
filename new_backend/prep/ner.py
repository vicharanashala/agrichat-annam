import pandas as pd
from gliner import GLiNER
import os


def merge_entities(entities, text):
    """
    Merges adjacent entities of the same type.
    
    Args:
        entities (list): A list of entity dictionaries from the model.
        text (str): The original text on which NER was performed.
    """
    if not entities:
        return []
    
    # Sort entities by their start position to ensure correct merging order
    entities = sorted(entities, key=lambda x: x['start'])
    
    merged = []
    current = entities[0]
    
    for next_entity in entities[1:]:
        # Check if the next entity is the same label and immediately follows the current one
        # (allowing for a single space or no space in between)
        if next_entity['label'] == current['label'] and next_entity['start'] <= current['end'] + 1:
            # Extend the end of the current entity
            current['end'] = next_entity['end']
            # Update the text to span the new, merged entity
            current['text'] = text[current['start']:current['end']].strip()
        else:
            merged.append(current)
            current = next_entity
            
    # Append the last processed entity
    merged.append(current)
    return merged


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


def main():
    print("Loading GLiNER model...")
    # Using a smaller, faster model for the example. Replace with your preferred one.
    model = GLiNER.from_pretrained("numind/NuZero_token")
    # model = GLiNER.from_pretrained("urchade/gliner_base") # Another good option
    print("Model loaded.")

    # Import and prepare labels
    from ner_labels import labels
    labels = [l.lower() for l in labels]

    data_path = "data/kcc_dataset_part_1.csv"
    chunk_size = 50  

    try:
        csv_reader = pd.read_csv(data_path, chunksize=chunk_size, iterator=True)
        print(f"Reading '{data_path}' in chunks of {chunk_size} rows...")

        for i, chunk in enumerate(csv_reader):
            print(f"\n--- Processing Chunk {i+1} ---")

            text_chunk = format_chunk_to_text(chunk)

            if not text_chunk.strip():
                print("Skipping empty chunk.")
                continue

            # print("Formatted Text for this chunk:\n", text_chunk) # Uncomment to debug
            
            print("Running NER model on the formatted text...")

            entities = model.predict_entities(text_chunk, labels, threshold=0.4)
            merged = merge_entities(entities, text_chunk)

            if not merged:
                print("No entities found in this chunk.")
            else:
                for entity in merged:
                    # Clean up the text from potential newlines before printing
                    clean_text = entity["text"].replace('\n', ' ')
                    print(f'  "{clean_text}" => {entity["label"]}')

    except FileNotFoundError:
        print(f"Error: The file '{data_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
