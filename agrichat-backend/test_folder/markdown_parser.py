import pathlib
import os
import pymupdf4llm 

class PDFtoMarkdownConverter:
    def __init__(self, input_folder, output_folder):
        self.input_folder = pathlib.Path(input_folder)
        self.output_folder = pathlib.Path(output_folder)
        
        # Ensure the output folder exists
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def convert(self):
        # Ensure the input folder exists
        if not self.input_folder.exists():
            print(f"Folder '{self.input_folder}' does not exist.")
            return
        
        # Loop through all PDF files in the folder
        for pdf_file in self.input_folder.glob("*.pdf"):
            try:
                # Convert PDF to markdown
                md_text = pymupdf4llm.to_markdown(str(pdf_file))
                
                # Define output markdown file path
                md_file = self.output_folder / (pdf_file.stem + ".md")
                
                # Save the markdown content
                md_file.write_bytes(md_text.encode())
                print(f"Converted: {pdf_file.name} -> {md_file.name}")
            except Exception as e:
                print(f"Failed to convert {pdf_file.name}: {e}")

# Example usage
if __name__ == "__main__":
    folder_path = r"C:\Users\amank\Downloads\RAG Dummy_v2\data\pdfs"
    output_folder = r"C:\Users\amank\Downloads\RAG Dummy_v2\markdown_files"
    converter = PDFtoMarkdownConverter(folder_path, output_folder)
    converter.convert()
