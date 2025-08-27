import os
import pandas as pd



def get_chunks(chunk_size):
  # Open all the .csv files in ./data/
  files = [f for f in os.listdir('./data/') if f.endswith('.csv')]
  chunks = []

  for file in files:
    file_path = os.path.join('./data/', file)
    df = pd.read_csv(file_path, chunksize=chunk_size)

    for chunk in df:
      chunk_text = chunk.to_csv(index=False, header=False)
      chunks.append(chunk_text)

  return chunks


def main():
  chunk_size = 5  # Define the chunk size
  chunks = get_chunks(chunk_size)


if __name__ == "__main__":
  main()


