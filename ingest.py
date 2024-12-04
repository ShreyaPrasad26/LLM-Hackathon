import json
import os
from typing import List, Dict
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
from tqdm import tqdm

logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for detailed logs
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)

def load_product_data(data_dir: str) -> List[Dict]:
    """Load product data from JSON files."""
    logging.debug(f"Starting to load product data from directory: {data_dir}")
    products = []
    files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    for filename in tqdm(files, desc="Loading product data"):
        file_path = os.path.join(data_dir, filename)
        try:
            with open(file_path, 'r') as f:
                product = json.load(f)
                products.append(product)
                logging.debug(f"Loaded product: {product.get('PRODUCT_NAME', 'Unknown')} from file: {filename}")
        except json.JSONDecodeError as e:
            logging.warning(f"Failed to parse JSON in {filename}: {str(e)}")
        except Exception as e:
            logging.error(f"Unexpected error while loading {filename}: {str(e)}")
    logging.info(f"Total products loaded: {len(products)}")
    return products

def flush_chroma_db(persist_directory: str) -> None:
    """Flush existing ChromaDB vector store."""
    logging.info("Flushing ChromaDB vector store...")
    if os.path.exists(persist_directory):
        for root, dirs, files in os.walk(persist_directory, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
                logging.debug(f"Deleted file: {file}")
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
                logging.debug(f"Deleted directory: {dir}")
        logging.info("ChromaDB vector store flushed successfully.")
    else:
        logging.info("No existing ChromaDB store found. Starting fresh.")

def prepare_documents(products: List[Dict]) -> List[Document]:
    """Prepare documents for vector store with text splitting."""
    logging.info("Starting document preparation...")
    documents = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for product in tqdm(products, desc="Processing products"):
        product_name = product.get('PRODUCT_NAME', 'Unknown Product')
        logging.debug(f"Processing product: {product_name}")
        
        fields = {
            "Description": product.get('DESCRIPTION', ''),
            "Clinical Pharmacology": product.get('CLINICAL PHARMACOLOGY', ''),
            "Indications and Usage": product.get('INDICATIONS AND USAGE', ''),
            "Contraindications": product.get('CONTRAINDICATIONS', ''),
            "Warnings": product.get('WARNINGS', ''),
            "Precautions": product.get('PRECAUTIONS', ''),
            "Adverse Reactions": product.get('ADVERSE REACTIONS', ''),
            "Dosage and Administration": product.get('DOSAGE AND ADMINISTRATION', ''),
        }

        for field_name, content in fields.items():
            if isinstance(content, str) and content.strip():
                logging.debug(f"Splitting content from field: {field_name} for product: {product_name}")
                chunks = splitter.split_text(content.strip())
                for chunk in chunks:
                    logging.debug(f"Created chunk for field: {field_name}, length: {len(chunk)}")
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "product_name": product_name,
                            "section": field_name,
                            "type": "pharmaceutical_info"
                        }
                    )
                    documents.append(doc)
            else:
                logging.debug(f"Skipping empty or invalid field: {field_name} for product: {product_name}")

    logging.info(f"Document preparation completed. Total chunks: {len(documents)}")
    return documents

def main():
    persist_directory = "./chroma_db"
    data_dir = "microlabs_usa_full_clean"
    
    # Step 1: Flush existing vector store
    flush_chroma_db(persist_directory)
    
    # Step 2: Load and prepare documents
    logging.info(f"Loading product data from {data_dir}")
    products = load_product_data(data_dir)
    
    if not products:
        logging.error("No products loaded. Check the data directory and files.")
        return
        
    documents = prepare_documents(products)
    
    if not documents:
        logging.error("No valid documents prepared. Check the product data structure.")
        return
        
    # Step 3: Create vector store
    logging.info("Creating embeddings and vector store...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    try:
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        logging.info(f"Vector store created successfully with {len(documents)} chunks at {persist_directory}")
    except Exception as e:
        logging.error(f"Failed to create or persist vector store: {str(e)}")

if __name__ == "__main__":
    main()
