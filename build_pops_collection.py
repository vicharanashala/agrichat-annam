#!/usr/bin/env python3
"""
Simple script to build the Package of Practices ChromaDB collection
using the cleaned markdown files from Extracted_digital_English_POP_data_md
"""

import sys
import os

# Add current directory to Python path
sys.path.append('/home/ubuntu/agrichat-annam')

try:
    from chroma_pops_builder import PoPsChromaBuilder
except ImportError as e:
    print(f"Error importing PoPsChromaBuilder: {e}")
    print("Make sure you're running this from the agrichat-annam directory")
    sys.exit(1)

def main():
    """Build the Package of Practices ChromaDB collection."""
    
    # Set paths
    chroma_path = "/home/ubuntu/agrichat-annam/agrichat-backend/chromaDb"
    pops_path = "/home/ubuntu/agrichat-annam/Extracted_digital_English_POP_data_md"
    
    print("Package of Practices ChromaDB Builder")
    print("=" * 60)
    print(f"ChromaDB Path: {chroma_path}")
    print(f"PoPs Data Path: {pops_path}")
    print("=" * 60)
    
    # Check if paths exist
    if not os.path.exists(chroma_path):
        print(f"ChromaDB directory does not exist: {chroma_path}")
        sys.exit(1)
    
    if not os.path.exists(pops_path):
        print(f"PoPs data directory does not exist: {pops_path}")
        sys.exit(1)
    
    # Initialize builder
    try:
        builder = PoPsChromaBuilder(chroma_path, pops_path)
    except Exception as e:
        print(f"Error initializing builder: {e}")
        sys.exit(1)
    
    # Build collection
    print("\n🚀 Starting collection build...")
    success = builder.build_collection()
    
    if success:
        print("\n🎉 Collection build completed successfully!")
        
        # Show stats
        try:
            stats = builder.get_collection_stats()
            print("\n📊 Collection Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        except Exception as e:
            print(f"⚠ Could not retrieve stats: {e}")
    else:
        print("\n Collection build failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()