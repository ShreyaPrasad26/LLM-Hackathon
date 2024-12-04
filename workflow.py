from typing import Annotated, Sequence, TypedDict, Union
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
import operator
import logging

logging.basicConfig(level=logging.DEBUG)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_step: str
    query: str
    result: Union[str, None]

def create_agent_workflow(assistant):
    # Define the function mapping
    functions = {
        "product_knowledge": assistant._query_product_database,
        "recommender": assistant._recommend_medicines,
        "alternatives": assistant._generate_alternatives,
        "summarizer": assistant._summarize_medicine,
        "internet_search": assistant.search_tool.run,
    }

    router_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a workflow router for a pharmaceutical assistant.
        Your task is to select the most appropriate tool for handling user queries.
        
        Available tools and their uses:
        1. product_knowledge - For questions about specific medicines or their details
        2. recommender - For requests seeking medicine recommendations based on symptoms
        3. alternatives - For queries about alternative medications
        4. summarizer - For summarizing detailed medicine information
        5. internet_search - Only if no relevant information can be found in the database
        
        You must respond with exactly one of these tool names:
        product_knowledge
        recommender
        alternatives
        summarizer
        internet_search
        
        Respond with only the tool name, no other text or explanation."""),
        ("human", "{query}")
    ])

    def route_query(state: AgentState) -> AgentState:
        query = state["query"]
        try:
            # Get raw response and clean it
            response = assistant.llm.invoke(router_prompt.format(query=query))
            cleaned_response = response.strip().lower()
            logging.debug(f"Raw router response: '{response}'")
            logging.debug(f"Cleaned response: '{cleaned_response}'")
            
            # Validate the response
            valid_tools = {
                "product_knowledge",
                "recommender",
                "alternatives",
                "summarizer",
                "internet_search",
            }
            
            if cleaned_response not in valid_tools:
                logging.warning(f"Invalid tool name received: '{cleaned_response}'. Defaulting to 'product_knowledge'")
                cleaned_response = "product_knowledge"
            
            state["next_step"] = cleaned_response
            logging.info(f"Selected tool: {cleaned_response}")
            
        except Exception as e:
            logging.error(f"Error in route_query: {str(e)}")
            state["next_step"] = "product_knowledge"  # Default fallback
            
        return state

    def execute_tool(state: AgentState) -> AgentState:
        tool_name = state["next_step"]
        query = state["query"]

        try:
            # Execute the selected tool
            logging.debug(f"Executing tool: {tool_name} for query: {query}")
            result = functions[tool_name](query)

            # Save the result
            state["result"] = result
            state["messages"].append(AIMessage(content=result))
        except Exception as e:
            logging.error(f"Error executing {tool_name}: {str(e)}")
            state["result"] = f"Error executing {tool_name}: {str(e)}"
            state["messages"].append(AIMessage(content=state["result"]))

        return state

    workflow = StateGraph(AgentState)
    workflow.add_node("route", route_query)
    workflow.add_node("execute", execute_tool)
    workflow.add_edge("route", "execute")
    workflow.add_edge("execute", END)
    workflow.set_entry_point("route")

    return workflow.compile()

def process_with_workflow(assistant, query: str):
    workflow = create_agent_workflow(assistant)

    state = {
        "messages": [HumanMessage(content=query)],
        "next_step": "",
        "query": query,
        "result": None,
    }

    result = workflow.invoke(state)
    return result["result"]
