import pandas as pd
import os

def process_csv_in_chunks(file_path, chunk_size=10):
    """
    Reads a CSV file in chunks, converts each chunk to a natural language format,
    and processes it.

    Args:
        file_path (str): The path to the CSV file.
        chunk_size (int): The number of rows per chunk.
    """
    try:
        # Read the header
        header = pd.read_csv(file_path, nrows=0).columns.tolist()

        # Process the rest of the file in chunks
        chunk_iter = pd.read_csv(file_path, skiprows=1, chunksize=chunk_size, header=None, names=header)

        for chunk in chunk_iter:
            natural_language_chunk = []
            for index, row in chunk.iterrows():
                row_text = []
                for col_name, col_value in row.items():
                    row_text.append(f"{col_name}: {col_value}")
                natural_language_chunk.append("; ".join(row_text))
            
            # Process the chunk (for now, just print it)
            print("--- New Chunk ---")
            for line in natural_language_chunk:
                print(line)

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    """
    Main function to demonstrate processing a CSV file.
    """
    file_path = os.path.join('data', 'kcc_dataset_part_1.csv')
    process_csv_in_chunks(file_path)

if __name__ == "__main__":
    main()
