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
import logging

logging.basicConfig(level=logging.DEBUG)

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
                json={"prompt": prompt, "max_tokens": 2000, "temperature": 0.7, "top_p": 1},
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            response_data = response.json()
            if 'choices' in response_data and len(response_data['choices']) > 0:
                return response_data['choices'][0].get('text', "No text returned.")
            return "Unexpected response format."
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"

    @property
    def _llm_type(self) -> str:
        return "LMStudio"

class PharmaKnowledgeAssistant:
    def __init__(self):
        self.llm = LMStudioLLM()
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=self.embeddings
        )
        self.search_tool = DuckDuckGoSearchRun()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.tools = self._initialize_tools()
        self.agent = self._initialize_agent()

    def _initialize_tools(self) -> List[Tool]:
        return [
            Tool(
                name="Product Knowledge Base",
                func=self._query_product_database,
                description="Answer questions about pharmaceutical products."
            ),
            Tool(
                name="Medicine Recommender",
                func=self._recommend_medicines,
                description="Get medicine recommendations based on symptoms."
            ),
            Tool(
                name="Alternative Medicines",
                func=self._generate_alternatives,
                description="Find alternative medications."
            ),
            Tool(
                name="Medicine Summarizer",
                func=self._summarize_medicine,
                description="Summarize medicine information."
            ),
            Tool(
                name="Internet Search",
                func=self.search_tool.run,
                description="Perform general medical searches."
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
        logging.debug(f"Querying product database: {query}")
        try:
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )
            result = qa_chain.invoke(query)
            logging.debug(f"Database query result: {result}")
            return result["result"]
        except Exception as e:
            logging.error(f"Error querying product database: {str(e)}")
            return f"Error querying product database: {str(e)}"

    def _recommend_medicines(self, query: str) -> str:
        template = """
        Based on the following query: {query}
        Provide medicine recommendations considering:
        - Symptoms and conditions
        - Common treatments
        - Safety considerations
        """
        prompt = PromptTemplate(template=template, input_variables=["query"])
        try:
            response = self.llm(prompt.format(query=query))
            logging.debug(f"Medicine recommendation: {response}")
            return response
        except Exception as e:
            logging.error(f"Error recommending medicines: {str(e)}")
            return f"Error generating recommendations: {str(e)}"

    def _generate_alternatives(self, medicine: str) -> str:
        template = """
        For the medicine: {medicine}
        Provide:
        - Alternative medications in the same class
        - Generic alternatives
        - Key differences between alternatives
        """
        prompt = PromptTemplate(template=template, input_variables=["medicine"])
        try:
            response = self.llm(prompt.format(medicine=medicine))
            logging.debug(f"Alternative medicines: {response}")
            return response
        except Exception as e:
            logging.error(f"Error generating alternatives: {str(e)}")
            return f"Error generating alternatives: {str(e)}"

    def _summarize_medicine(self, medicine_info: str) -> str:
        template = """
        Provide a concise summary of the following medicine information:
        {medicine_info}
        Include:
        - Key uses
        - Important warnings
        - Main side effects
        - Essential dosing information
        """
        prompt = PromptTemplate(template=template, input_variables=["medicine_info"])
        try:
            response = self.llm(prompt.format(medicine_info=medicine_info))
            logging.debug(f"Medicine summary: {response}")
            return response
        except Exception as e:
            logging.error(f"Error summarizing medicine: {str(e)}")
            return f"Error generating summary: {str(e)}"

    def process_query(self, query: str) -> str:
        try:
            from workflow import process_with_workflow
            return process_with_workflow(self, query)
        except Exception as e:
            logging.error(f"Error in process_query: {str(e)}")
            return f"An error occurred: {str(e)}"
