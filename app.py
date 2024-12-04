# app.py
import os
from typing import List, Dict, Any
from langchain.agents import Tool, AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.language_models import LLM
from langchain.chains import RetrievalQA
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_community.llms import OpenAI

# Custom LLM class for LM Studio
class LMStudioLLM(LLM):
    base_url: str = "http://localhost:1234/v1"
    client: Any = None
    session: Any = None
    
    def __init__(self):
        super().__init__()
        import requests
        self.session = requests.Session()
        
    def _call(self, prompt: str, **kwargs) -> str:
        try:
            response = self.session.post(
                f"{self.base_url}/completions",
                json={
                    "prompt": prompt,
                    "max_tokens": 2000,
                    "temperature": 0.7,
                    "top_p": 1,
                    "stream": False
                },
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()['choices'][0]['text']
        except Exception as e:
            return f"Error generating response: {str(e)}"

    @property
    def _llm_type(self) -> str:
        return "LMStudio"

class PharmaKnowledgeAssistant:
    def __init__(self):
        # Initialize LLM using LMStudio
        self.llm = LMStudioLLM()
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize ChromaDB
        self.vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=self.embeddings
        )
        
        # Initialize search tool
        self.search_tool = DuckDuckGoSearchRun()
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize tools
        self.tools = self._initialize_tools()
        
        # Initialize agent
        self.agent = self._initialize_agent()

    def _initialize_tools(self) -> List[Tool]:
        return [
            Tool(
                name="Product Knowledge Base",
                func=self._query_product_database,
                description="Use this tool for answering questions about pharmaceutical products using our database"
            ),
            Tool(
                name="Medicine Recommender",
                func=self._recommend_medicines,
                description="Use this tool to get personalized medicine recommendations based on symptoms and conditions"
            ),
            Tool(
                name="Alternative Medicines",
                func=self._generate_alternatives,
                description="Use this tool to find alternative medications for a given medicine"
            ),
            Tool(
                name="Medicine Summarizer",
                func=self._summarize_medicine,
                description="Use this tool to get concise summaries of medicine information"
            ),
            Tool(
                name="Internet Search",
                func=self.search_tool.run,
                description="Use this tool when information is not available in our database"
            )
        ]

    def _initialize_agent(self) -> AgentExecutor:
        return initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True
        )

    def _query_product_database(self, query: str) -> str:
        try:
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )
            result = qa_chain.invoke(query)
            return result["result"]
        except Exception as e:
            return f"Error querying product database: {str(e)}"

    def _recommend_medicines(self, query: str) -> str:
        template = """
        Based on the following query: {query}
        Please provide medicine recommendations considering:
        1. Symptoms and conditions described
        2. Common treatments for these conditions
        3. Safety considerations
        
        Provide a structured recommendation with reasoning.
        """
        prompt = PromptTemplate(template=template, input_variables=["query"])
        try:
            return self.llm(prompt.format(query=query))
        except Exception as e:
            return f"Error generating recommendations: {str(e)}"

    def _generate_alternatives(self, medicine: str) -> str:
        template = """
        For the medicine: {medicine}
        Please provide:
        1. Alternative medications in the same class
        2. Generic alternatives if applicable
        3. Key differences between alternatives
        """
        prompt = PromptTemplate(template=template, input_variables=["medicine"])
        try:
            return self.llm(prompt.format(medicine=medicine))
        except Exception as e:
            return f"Error generating alternatives: {str(e)}"

    def _summarize_medicine(self, medicine_info: str) -> str:
        template = """
        Please provide a concise summary of the following medicine information:
        {medicine_info}
        
        Include:
        1. Key uses
        2. Important warnings
        3. Main side effects
        4. Essential dosing information
        """
        prompt = PromptTemplate(template=template, input_variables=["medicine_info"])
        try:
            return self.llm(prompt.format(medicine_info=medicine_info))
        except Exception as e:
            return f"Error generating summary: {str(e)}"

    def process_query(self, query: str) -> str:
        """Main method to process user queries using the workflow"""
        try:
            from workflow import process_with_workflow
            return process_with_workflow(self, query)
        except Exception as e:
            return f"An error occurred: {str(e)}"
