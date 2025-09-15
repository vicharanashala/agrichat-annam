"""
Package of Practices (PoPs) ChromaDB Builder

This script builds a ChromaDB collection from Package of Practices JSON files,
creating a separate collection alongside the existing agricultural knowledge base.
"""

import os
import json
import sys
from typing import List, Dict, Any
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from local_llm_interface import local_embeddings
import argparse
from datetime import datetime


class PoPsChromaBuilder:
    """Builder class for Package of Practices ChromaDB collection."""
    
    def __init__(self, chroma_path: str, pops_data_path: str):
        """
        Initialize the PoPs ChromaDB builder.
        
        Args:
            chroma_path: Path to ChromaDB directory
            pops_data_path: Path to Package_of_practices directory
        """
        self.chroma_path = chroma_path
        self.pops_data_path = pops_data_path
        self.collection_name = "package_of_practices"
        self.embeddings = local_embeddings
        
        print(f"[PoPs Builder] ChromaDB path: {chroma_path}")
        print(f"[PoPs Builder] PoPs data path: {pops_data_path}")
        print(f"[PoPs Builder] Collection name: {self.collection_name}")
    
    def extract_text_from_json(self, json_data: List[Dict], filename: str) -> List[str]:
        """
        Extract text content from Package of Practices JSON structure.
        
        Args:
            json_data: JSON data from PoPs file
            filename: Name of the source file
            
        Returns:
            List of text chunks extracted from the JSON
        """
        text_chunks = []
        
        for section in json_data:
            heading = section.get('heading', '')
            content_list = section.get('content', [])
            
            # Create a section header
            section_text = f"## {heading}\n\n"
            
            for content_item in content_list:
                content_type = content_item.get('type', '')
                
                if content_type == 'text':
                    section_text += content_item.get('content', '') + "\n\n"
                
                elif content_type == 'ordered_list' or content_type == 'unordered_list':
                    list_items = content_item.get('content', [])
                    for i, item in enumerate(list_items, 1):
                        if isinstance(item, dict):
                            text = item.get('text', '')
                            link = item.get('link', '')
                            if content_type == 'ordered_list':
                                section_text += f"{i}. {text}"
                            else:
                                section_text += f"• {text}"
                            if link:
                                section_text += f" (Reference: {link})"
                            section_text += "\n"
                        else:
                            section_text += f"• {item}\n"
                    section_text += "\n"
                
                elif content_type == 'table':
                    # Handle table content
                    table_data = content_item.get('content', {})
                    headers = table_data.get('headers', [])
                    rows = table_data.get('rows', [])
                    
                    if headers:
                        section_text += " | ".join(headers) + "\n"
                        section_text += "|".join(["---"] * len(headers)) + "\n"
                    
                    for row in rows:
                        section_text += " | ".join(str(cell) for cell in row) + "\n"
                    section_text += "\n"
            
            # Add metadata about source
            crop_category = self._extract_crop_category(filename)
            section_text += f"\n[Source: Package of Practices - {crop_category}]"
            
            text_chunks.append(section_text.strip())
        
        return text_chunks
    
    def _extract_crop_category(self, filename: str) -> str:
        """Extract crop category from file path."""
        # Extract category from path like /Cereals_and_millets/Maize.json
        path_parts = filename.replace('\\', '/').split('/')
        
        if len(path_parts) >= 2:
            category = path_parts[-2].replace('_', ' ').title()
            crop = path_parts[-1].replace('.json', '').replace('_', ' ').title()
            return f"{category} - {crop}"
        else:
            crop = filename.replace('.json', '').replace('_', ' ').title()
            return f"General - {crop}"
    
    def process_pops_files(self) -> List[Document]:
        """
        Process all Package of Practices JSON files and create Documents.
        
        Returns:
            List of Document objects for ChromaDB ingestion
        """
        documents = []
        
        if not os.path.exists(self.pops_data_path):
            print(f"[ERROR] PoPs data path not found: {self.pops_data_path}")
            return documents
        
        # Walk through all JSON files in the PoPs directory
        total_files = 0
        processed_files = 0
        
        for root, dirs, files in os.walk(self.pops_data_path):
            json_files = [f for f in files if f.endswith('.json')]
            total_files += len(json_files)
            
            for filename in json_files:
                file_path = os.path.join(root, filename)
                relative_path = os.path.relpath(file_path, self.pops_data_path)
                
                try:
                    print(f"[PoPs Builder] Processing: {relative_path}")
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                    
                    # Extract text chunks from JSON
                    text_chunks = self.extract_text_from_json(json_data, relative_path)
                    
                    # Create documents for each chunk
                    for i, text_chunk in enumerate(text_chunks):
                        if text_chunk.strip():  # Only add non-empty chunks
                            doc = Document(
                                page_content=text_chunk,
                                metadata={
                                    'source': 'PoPs database',
                                    'file_path': relative_path,
                                    'crop_category': self._extract_crop_category(relative_path),
                                    'chunk_id': i,
                                    'total_chunks': len(text_chunks),
                                    'processed_date': datetime.now().isoformat()
                                }
                            )
                            documents.append(doc)
                    
                    processed_files += 1
                    
                except Exception as e:
                    print(f"[ERROR] Failed to process {relative_path}: {e}")
                    continue
        
        print(f"[PoPs Builder] Processed {processed_files}/{total_files} files")
        print(f"[PoPs Builder] Created {len(documents)} document chunks")
        
        return documents
    
    def build_collection(self) -> bool:
        """
        Build the Package of Practices ChromaDB collection.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"[PoPs Builder] Starting PoPs collection build...")
            
            # Process all PoPs files
            documents = self.process_pops_files()
            
            if not documents:
                print(f"[ERROR] No documents to process")
                return False
            
            # Create ChromaDB collection
            print(f"[PoPs Builder] Creating ChromaDB collection: {self.collection_name}")
            
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                persist_directory=self.chroma_path
            )
            
            # Persist the collection
            vectorstore.persist()
            
            print(f"[PoPs Builder] Successfully built PoPs collection with {len(documents)} documents")
            print(f"[PoPs Builder] Collection saved to: {self.chroma_path}")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to build PoPs collection: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the PoPs collection.
        
        Returns:
            Dict containing collection statistics
        """
        try:
            vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.chroma_path
            )
            
            collection = vectorstore._collection
            stats = {
                'collection_name': self.collection_name,
                'total_documents': collection.count(),
                'created_date': datetime.now().isoformat()
            }
            
            return stats
            
        except Exception as e:
            print(f"[ERROR] Failed to get collection stats: {e}")
            return {'error': str(e)}


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Build Package of Practices ChromaDB Collection')
    parser.add_argument('--chroma-path', type=str, required=True, 
                       help='Path to ChromaDB directory')
    parser.add_argument('--pops-path', type=str, required=True,
                       help='Path to Package_of_practices directory')
    parser.add_argument('--stats', action='store_true',
                       help='Show collection statistics after building')
    
    args = parser.parse_args()
    
    # Build the collection
    builder = PoPsChromaBuilder(args.chroma_path, args.pops_path)
    success = builder.build_collection()
    
    if success:
        print(f"\n[SUCCESS] Package of Practices collection built successfully!")
        
        if args.stats:
            stats = builder.get_collection_stats()
            print(f"\nCollection Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
    else:
        print(f"\n[FAILED] Package of Practices collection build failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
