from markdown_parser import PDFtoMarkdownConverter
from creating_database import ChromaDataStore
from query_data import ChromaQueryHandler
import os
import shutil

pdf_folder = r"C:\Users\amank\Downloads\RAG Dummy_v2\data\pdfs"
markdown_folder = r"C:\Users\amank\Downloads\RAG Dummy_v2\data\pdfs"
converted_pdfs_folder = r"C:\Users\amank\Downloads\RAG Dummy_v2\converted_pdfs"
chroma_path = "chroma"

os.makedirs(converted_pdfs_folder, exist_ok=True)

pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

if pdf_files:
    print("New PDF files found. Converting to Markdown...")
    
    pdf_converter = PDFtoMarkdownConverter(pdf_folder, markdown_folder)
    pdf_converter.convert()
    
    for pdf_file in pdf_files:
        src_path = os.path.join(pdf_folder, pdf_file)
        dest_path = os.path.join(converted_pdfs_folder, pdf_file)
        shutil.move(src_path, dest_path)
        print(f"Moved {pdf_file} to {converted_pdfs_folder}.")
else:
    print("No new PDF files found. Skipping conversion.")

if os.path.exists(chroma_path) and os.listdir(chroma_path) and not(pdf_files):
    print("ChromaDB already contains data. Skipping data store creation.")
else:
    print("Generating ChromaDB data store...")
    chroma_store = ChromaDataStore(markdown_folder, chroma_path)
    chroma_store.generate_data_store()

query_text = input("Enter your query: ")
query_handler = ChromaQueryHandler(chroma_path)
query_handler.search_query(query_text)