# ingest.py
import json
import os
from typing import List, Dict
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

def load_product_data(data_dir: str) -> List[Dict]:
    """Load product data from JSON files"""
    products = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            with open(os.path.join(data_dir, filename), 'r') as f:
                product = json.load(f)
                products.append(product)
    return products

def prepare_documents(products: List[Dict]) -> List[Document]:
    """Prepare documents for vector store"""
    documents = []
    for product in products:
        # Clean up duplicated content by taking first instance
        cleaned_content = {
            key: value.split('\n')[0] if isinstance(value, str) else value 
            for key, value in product.items()
        }
        
        # Get product name or use a default value
        product_name = cleaned_content.get('product_name') or cleaned_content.get('PRODUCT NAME') or 'Unknown Product'
        
        # Create a comprehensive document for each product
        content = f"""
Product: {product_name}

Description: {cleaned_content.get('DESCRIPTION', '')}

Clinical Pharmacology: {cleaned_content.get('CLINICAL PHARMACOLOGY', '')}

Indications and Usage: {cleaned_content.get('INDICATIONS AND USAGE', '')}

Contraindications: {cleaned_content.get('CONTRAINDICATIONS', '')}

Warnings: {cleaned_content.get('WARNINGS', '')}

Precautions: {cleaned_content.get('PRECAUTIONS', '')}

Adverse Reactions: {cleaned_content.get('ADVERSE REACTIONS', '')}

Dosage and Administration: {cleaned_content.get('DOSAGE AND ADMINISTRATION', '')}
"""
        # Create a Document object
        doc = Document(
            page_content=content,
            metadata={
                'product_name': product_name,
                'type': 'product_info'
            }
        )
        documents.append(doc)
    return documents

def main():
    # Initialize embeddings with specific model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Load product data
    products = load_product_data("microlabs_usa_full_clean")
    
    # Prepare documents
    documents = prepare_documents(products)
    
    # Create and populate vector store
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    print(f"Ingested {len(documents)} documents into the vector store")

if __name__ == "__main__":
    main()