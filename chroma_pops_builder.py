"""
Package of Practices (PoPs) ChromaDB Builder

This script builds a ChromaDB collection from Package of Practices markdown files,
creating a separate collection alongside the existing agricultural knowledge base.
"""

import logging
import os
import sys
import re
from typing import List, Dict, Any, Tuple
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import argparse
from datetime import datetime
import chromadb

# Add the backend directory to the path to import local modules
sys.path.append('/home/ubuntu/agrichat-annam/agrichat-backend')
from pipeline.llm_adapter import local_embeddings


logger = logging.getLogger(__name__)
class PoPsChromaBuilder:
    """Builder class for Package of Practices ChromaDB collection."""
    
    def __init__(self, chroma_path: str, pops_data_path: str):
        """
        Initialize the PoPs ChromaDB builder.
        
        Args:
            chroma_path: Path to ChromaDB directory
            pops_data_path: Path to Extracted_digital_English_POP_data_md directory
        """
        logger.info(f"[CHROMA_POPS_BUILDER.PY] __init__() - INVOKED")
        self.chroma_path = chroma_path
        self.pops_data_path = pops_data_path
        self.collection_name = "package_of_practices"
        self.embeddings = local_embeddings
        
    
    def delete_existing_collection(self) -> bool:
        """
        Delete the existing package_of_practices collection if it exists.
        
        Returns:
            bool: True if deletion was successful or collection didn't exist, False otherwise
        """
        logger.info(f"[CHROMA_POPS_BUILDER.PY] delete_existing_collection() - INVOKED")
        try:
            import chromadb
            client = chromadb.PersistentClient(path=self.chroma_path)
            
            # Check if collection exists
            collections = client.list_collections()
            collection_names = [col.name for col in collections]
            
            if self.collection_name in collection_names:
                client.delete_collection(name=self.collection_name)
            return True
            
        except Exception as e:
            return False
    
    def extract_text_from_markdown(self, file_path: str) -> str:
        """
        Extract text content from Package of Practices markdown file.
        
        Args:
            file_path: Path to the markdown file
            
        Returns:
            Text content from the markdown file
        """
        logger.info(f"[CHROMA_POPS_BUILDER.PY] extract_text_from_markdown() - INVOKED")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Clean up the content - remove excessive blank lines
            lines = content.split('\n')
            cleaned_lines = []
            
            for line in lines:
                # Skip lines with only <!-- image --> comments
                if line.strip() == '<!-- image -->':
                    continue
                cleaned_lines.append(line)
            
            # Join back and normalize whitespace
            cleaned_content = '\n'.join(cleaned_lines)
            
            # Remove excessive blank lines (more than 2 consecutive)
            import re
            cleaned_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_content)
            
            return cleaned_content.strip()
            
        except Exception as e:
            logger.error(f"Error reading markdown file {file_path}: {e}")
            return ""
    
    def _extract_crop_category(self, filename: str) -> str:
        """Extract crop category from file path."""
        logger.info(f"[CHROMA_POPS_BUILDER.PY] _extract_crop_category() - INVOKED")
        path_parts = filename.replace('\\', '/').split('/')
        
        if len(path_parts) >= 2:
            category = path_parts[-2].replace('_', ' ').title()
            crop = path_parts[-1].replace('.md', '').replace('_', ' ').title()
            return f"{category} - {crop}"
        else:
            crop = filename.replace('.md', '').replace('_', ' ').title()
            return f"General - {crop}"
    
    def _extract_state_and_name(self, file_path: str) -> Tuple[str, str]:
        """Extract state and document name from markdown file path."""
        logger.info(f"[CHROMA_POPS_BUILDER.PY] _extract_state_and_name() - INVOKED")
        
        # Get relative path from the pops_data_path
        relative_path = os.path.relpath(file_path, self.pops_data_path)
        path_parts = relative_path.replace('\\', '/').split('/')
        
        # Extract state from the first directory (e.g., "Assam", "Bihar", etc.)
        if len(path_parts) >= 2:
            state = path_parts[0].replace('_', ' ').title()
            # Extract name from filename
            name = path_parts[-1].replace('.md', '').replace('_', ' ').replace('-', ' ').title()
        else:
            state = "Others"
            name = path_parts[0].replace('.md', '').replace('_', ' ').replace('-', ' ').title()
        
        return state, name
    
    def process_pops_files(self) -> List[Document]:
        """
        Process all Package of Practices markdown files and create Documents.
        
        Returns:
            List of Document objects for ChromaDB ingestion
        """
        logger.info(f"[CHROMA_POPS_BUILDER.PY] process_pops_files() - INVOKED")
        documents = []
        
        if not os.path.exists(self.pops_data_path):
            print(f"Error: Directory {self.pops_data_path} does not exist")
            return documents
        
        total_files = 0
        processed_files = 0
        
        print(f"Scanning directory: {self.pops_data_path}")
        
        for root, dirs, files in os.walk(self.pops_data_path):
            md_files = [f for f in files if f.endswith('.md')]
            total_files += len(md_files)
            
            for filename in md_files:
                file_path = os.path.join(root, filename)
                relative_path = os.path.relpath(file_path, self.pops_data_path)
                
                try:
                    print(f"Processing: {relative_path}")
                    
                    # Extract content from markdown file
                    content = self.extract_text_from_markdown(file_path)
                    
                    if content.strip():
                        # Extract state and name from file path
                        state, name = self._extract_state_and_name(file_path)
                        
                        # Add source information to content
                        content_with_source = f"{content}\n\n[Source: Package of Practices - {state} - {name}]"
                        
                        doc = Document(
                            page_content=content_with_source,
                            metadata={
                                'state': state,
                                'name': name,
                                'source_file': relative_path,
                                'content_type': 'package_of_practices'
                            }
                        )
                        documents.append(doc)
                        processed_files += 1
                        print(f"  ✓ Added document: {state} - {name}")
                    else:
                        print(f"  ⚠ Empty content in {relative_path}")
                        
                except Exception as e:
                    print(f"  ✗ Error processing {relative_path}: {e}")
                    continue
        
        print(f"\nProcessing complete:")
        print(f"  Total markdown files found: {total_files}")
        print(f"  Successfully processed: {processed_files}")
        print(f"  Documents created: {len(documents)}")
        
        return documents
    
    def build_collection(self) -> bool:
        """
        Build the Package of Practices ChromaDB collection.
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info(f"[CHROMA_POPS_BUILDER.PY] build_collection() - INVOKED")
        try:
            
            # First delete existing collection if it exists
            if not self.delete_existing_collection():
                return False
            
            documents = self.process_pops_files()
            
            if not documents:
                print("No documents were processed. Collection not created.")
                return False
            
            print(f"\nBuilding ChromaDB collection '{self.collection_name}' with {len(documents)} documents...")
            
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                persist_directory=self.chroma_path
            )
            
            # Persist the collection
            vectorstore.persist()
            
            print(f"✅ Successfully created collection '{self.collection_name}' with {len(documents)} documents")
            return True
            
        except Exception as e:
            return False
    
    def delete_collection(self) -> bool:
        """
        Delete the existing PoPs collection.
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info(f"[CHROMA_POPS_BUILDER.PY] delete_collection() - INVOKED")
        try:
            import chromadb
            client = chromadb.PersistentClient(path=self.chroma_path)
            
            # Check if collection exists
            collections = client.list_collections()
            collection_names = [col.name for col in collections]
            
            if self.collection_name in collection_names:
                client.delete_collection(name=self.collection_name)
                return True
            else:
                return True
                
        except Exception as e:
            return False

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the PoPs collection.
        
        Returns:
            Dict containing collection statistics
        """
        logger.info(f"[CHROMA_POPS_BUILDER.PY] get_collection_stats() - INVOKED")
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
            return {'error': str(e)}


logger = logging.getLogger(__name__)
def main():
    """Main function for command-line usage."""
    logger.info(f"[CHROMA_POPS_BUILDER.PY] main() - INVOKED")
    parser = argparse.ArgumentParser(description='Build Package of Practices ChromaDB Collection')
    parser.add_argument('--chroma-path', type=str, required=True, 
                       help='Path to ChromaDB directory')
    parser.add_argument('--pops-path', type=str, required=True,
                       help='Path to Extracted_digital_English_POP_data_md directory')
    parser.add_argument('--stats', action='store_true',
                       help='Show collection statistics after building')
    
    args = parser.parse_args()
    
    builder = PoPsChromaBuilder(args.chroma_path, args.pops_path)
    success = builder.build_collection()
    
    if success:
        
        if args.stats:
            stats = builder.get_collection_stats()
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
