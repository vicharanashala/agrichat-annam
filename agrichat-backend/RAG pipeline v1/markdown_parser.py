import pathlib
import os
import pymupdf4llm 

class PDFtoMarkdownConverter:
    def __init__(self, input_folder, output_folder):
        self.input_folder = pathlib.Path(input_folder)
        self.output_folder = pathlib.Path(output_folder)
        
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def convert(self):
        if not self.input_folder.exists():
            print(f"Folder '{self.input_folder}' does not exist.")
            return
        
        for pdf_file in self.input_folder.glob("*.pdf"):
            try:
                md_text = pymupdf4llm.to_markdown(str(pdf_file))
                
                md_file = self.output_folder / (pdf_file.stem + ".md")
                
                md_file.write_bytes(md_text.encode())
                print(f"Converted: {pdf_file.name} -> {md_file.name}")
            except Exception as e:
                print(f"Failed to convert {pdf_file.name}: {e}")

if __name__ == "__main__":
    folder_path = r"C:\Users\amank\Downloads\RAG Dummy_v2\data" 
    output_folder = r"C:\Users\amank\Downloads\RAG Dummy_v2\markdown_files"
    converter = PDFtoMarkdownConverter(folder_path, output_folder)
    converter.convert()
