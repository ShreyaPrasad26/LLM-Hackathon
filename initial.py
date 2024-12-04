#pip install chromadb langchain

import os
import json
import chromadb
from langchain.agents import Tool, initialize_agent
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import LLAMA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# Function to load JSON files from a directory
def load_json_from_directory(directory_path):
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, "r") as file:
                data = json.load(file)
                # Process each file into a text block
                for product in data:
                    combined_text = (
                        f"Product Name: {product.get('product_name', 'N/A')}\n"
                        f"Description: {product.get('DESCRIPTION:', 'N/A')}\n"
                        f"Pharmacology: {product.get('CLINICAL PHARMACOLOGY:', 'N/A')}\n"
                        f"Indications: {product.get('INDICATIONS AND USAGE:', 'N/A')}\n"
                        f"Contraindications: {product.get('CONTRAINDICATIONS:', 'N/A')}\n"
                        f"Warnings: {product.get('WARNINGS:', 'N/A')}\n"
                        f"Precautions: {product.get('PRECAUTIONS:', 'N/A')}\n"
                        f"Adverse Reactions: {product.get('ADVERSE REACTIONS:', 'N/A')}\n"
                        f"Overdosage: {product.get('OVERDOSAGE:', 'N/A')}\n"
                        f"Dosage: {product.get('DOSAGE AND ADMINISTRATION:', 'N/A')}\n"
                        f"How Supplied: {product.get('HOW SUPPLIED:', 'N/A')}\n"
                        f"Ingredients and Appearance: {product.get('INGREDIENTS AND APPEARANCE', 'N/A')}\n"
                        f"Package Label: {product.get('PACKAGE LABEL.PRINCIPAL DISPLAY PANEL:', 'N/A')}\n"
                    )
                    documents.append(combined_text)
    return documents

# Step 1: Set up OpenAI Embeddings
embedding = OpenAIEmbeddings()

# Step 2: Initialize the Chroma vector store
client = chromadb.Client()
chroma_db = client.create_collection("pharma_products")

# Step 3: Load JSON data from a directory
directory_path = "path_to_your_json_directory"  # Change to your directory path
documents = load_json_from_directory(directory_path)

# Step 4: Create embeddings for each document and store them in ChromaDB
embeddings = [embedding.embed_query(doc) for doc in documents]
for doc, emb in zip(documents, embeddings):
    chroma_db.add([{
        'id': str(documents.index(doc)),
        'embedding': emb,
        'metadata': {'text': doc}
    }])

# Step 5: Create the LLAMA LLM for responses (replace with your LLAMA setup)
llama = LLAMA(model_name="your-llama-model")  # Replace with actual LLAMA model path

# Step 6: Define tools for QA, Summarization, and Recommendation
def qa_function(query: str):
    # Retrieve the most relevant documents using ChromaDB
    result = chroma_db.query(query, n_results=3)
    docs = "\n".join([item['metadata']['text'] for item in result['documents']])
    prompt = f"Answer the question: {query} using the following documents:\n{docs}"
    response = llama.generate([prompt])
    return response[0].strip()

qa_tool = Tool(
    name="QA Tool",
    func=qa_function,
    description="Answers questions based on the available product data"
)

def summarize_function(query: str):
    prompt = f"Summarize the following text: {query}"
    response = llama.generate([prompt])
    return response[0].strip()

summarization_tool = Tool(
    name="Summarizer",
    func=summarize_function,
    description="Summarizes the provided text"
)

def recommend_function(query: str):
    # Simple placeholder logic for recommendations based on query
    if "stomach ulcers" in query:
        alternatives = ["Paracetamol", "Aspirin"]
        response = f"Considering your condition, alternatives include: {', '.join(alternatives)}."
    else:
        response = "No contraindications found, recommend continuing with your current prescription."
    return response

recommendation_tool = Tool(
    name="Recommendation Tool",
    func=recommend_function,
    description="Recommend alternatives or related drugs"
)

# Initialize agent with tools
tools = [qa_tool, summarization_tool, recommendation_tool]
memory = ConversationBufferMemory()

agent = initialize_agent(tools=tools, agent_type="zero-shot-react-description", memory=memory, verbose=True)

# Command-line input handling
import argparse

def main(query):
    response = agent.run(query)
    print(f"Response: {response}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pharmaceutical Question Answering System")
    parser.add_argument("query", type=str, help="The question to ask about pharmaceutical products.")
    args = parser.parse_args()
    main(args.query)
