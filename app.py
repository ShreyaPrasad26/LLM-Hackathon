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
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
logging.basicConfig(level=logging.DEBUG)

class LMStudioLLM(LLM):
    base_url: str = "http://localhost:1234/v1"
    client: Any = None
    session: Any = None
    max_tokens: int = 4096
    tokenizer: Any = None
    timeout: int = 60
    max_retries: int = 3

    def __init__(self):
        super().__init__()
        import requests
        from transformers import AutoTokenizer
        self.session = requests.Session()
        
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    def _truncate_prompt(self, prompt: str) -> str:
        tokens = self.tokenizer.encode(prompt)
        if len(tokens) > (self.max_tokens - 1000):
            tokens = tokens[:(self.max_tokens - 1000)]
            prompt = self.tokenizer.decode(tokens)
        return prompt

    def _call(self, prompt: str, **kwargs) -> str:
        attempts = 0
        while attempts < self.max_retries:
            try:
                truncated_prompt = self._truncate_prompt(prompt)
                
                payload = {
                    "model": "local-model",
                    "prompt": truncated_prompt,
                    "max_tokens": 1000,
                    "temperature": 0.7,
                    "top_p": 1,
                    "stream": False,
                    "stop": None
                }
                
                response = self.session.post(
                    f"{self.base_url}/completions",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    if 'choices' in response_data and len(response_data['choices']) > 0:
                        return response_data['choices'][0].get('text', '').strip()
                    return "No text returned from model."
                else:
                    error_msg = f"API Error: {response.status_code} - {response.text}"
                    logging.error(error_msg)
                    
            except requests.exceptions.Timeout:
                attempts += 1
                if attempts < self.max_retries:
                    wait_time = 2 ** attempts
                    logging.warning(f"Request timed out. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                return "Request timed out after multiple retries. Please ensure LMStudio is not overloaded."
                
            except requests.exceptions.ConnectionError:
                return "Cannot connect to LMStudio. Please ensure it's running on http://localhost:1234"
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                logging.error(error_msg)
                return error_msg
            
            attempts += 1
            
        return "Maximum retry attempts reached. Please try again later."

    @property
    def _llm_type(self) -> str:
        return "LMStudio"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "base_url": self.base_url
        }

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
        try:
            # First query the vector database
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
            context = retriever.get_relevant_documents(query)
            context_text = "\n".join([doc.page_content for doc in context])
            
            template = """
            Using the following pharmaceutical knowledge base information:
            {context}
            
            Based on the query: {query}
            Provide medicine recommendations considering:
            - Symptoms and conditions
            - Common treatments
            - Safety considerations
            
            Only recommend medicines that are mentioned in the knowledge base.
            """
            prompt = PromptTemplate(template=template, input_variables=["context", "query"])
            response = self.llm(prompt.format(context=context_text, query=query))
            logging.debug(f"Medicine recommendation: {response}")
            return response
        except Exception as e:
            logging.error(f"Error recommending medicines: {str(e)}")
            return f"Error generating recommendations: {str(e)}"

    def _generate_alternatives(self, medicine: str) -> str:
        try:
            # Query vector database for the medicine and similar ones
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
            context = retriever.get_relevant_documents(medicine)
            context_text = "\n".join([doc.page_content for doc in context])
            
            template = """
            Using the following pharmaceutical knowledge base information:
            {context}
            
            For the medicine: {medicine}
            Provide:
            - Alternative medications in the same class
            - Generic alternatives
            - Key differences between alternatives
            
            Only suggest alternatives that are mentioned in the knowledge base.
            """
            prompt = PromptTemplate(template=template, input_variables=["context", "medicine"])
            response = self.llm(prompt.format(context=context_text, medicine=medicine))
            logging.debug(f"Alternative medicines: {response}")
            return response
        except Exception as e:
            logging.error(f"Error generating alternatives: {str(e)}")
            return f"Error generating alternatives: {str(e)}"

    def _summarize_medicine(self, medicine_info: str) -> str:
        try:
            # Query vector database for relevant information
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})
            context = retriever.get_relevant_documents(medicine_info)
            context_text = "\n".join([doc.page_content for doc in context])
            
            template = """
            Using the following pharmaceutical knowledge base information:
            {context}
            
            Provide a concise summary of this medicine:
            {medicine_info}
            
            Include:
            - Key uses
            - Important warnings
            - Main side effects
            - Essential dosing information
            
            Base the summary only on information present in the knowledge base.
            """
            prompt = PromptTemplate(template=template, input_variables=["context", "medicine_info"])
            response = self.llm(prompt.format(context=context_text, medicine_info=medicine_info))
            logging.debug(f"Medicine summary: {response}")
            return response
        except Exception as e:
            logging.error(f"Error summarizing medicine: {str(e)}")
            return f"Error generating summary: {str(e)}"

    def process_query(self, query: str) -> str:
        try:
            from workflow import process_with_workflow
            response = process_with_workflow(self, query)
            # Extract only the assistant's response
            if isinstance(response, str) and "Assistant:" in response:
                return response.split("Assistant:")[1].strip()
            return response
        except Exception as e:
            logging.error(f"Error in process_query: {str(e)}")
            return f"An error occurred: {str(e)}"
