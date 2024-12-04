from typing import Annotated, Sequence, TypedDict, Union
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
import operator

# Define state type
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_step: str
    query: str
    result: Union[str, None]

def create_agent_workflow(assistant):
    # Define the functions mapping without ToolExecutor
    functions = {
        "product_knowledge": assistant._query_product_database,
        "recommender": assistant._recommend_medicines,
        "alternatives": assistant._generate_alternatives,
        "summarizer": assistant._summarize_medicine,
        "internet_search": assistant.search_tool.run
    }

    router_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a workflow router for a pharmaceutical knowledge assistant.
        Based on the user's query, determine which tool would be most appropriate to use:
        
        - product_knowledge: For questions about specific medicines and their details
        - recommender: For requests seeking medicine recommendations based on symptoms
        - alternatives: For queries about alternative medications
        - summarizer: For requests to summarize medicine information
        - internet_search: For general medical queries not covered by other tools
        
        Respond with only the tool name that best matches the query."""),
        ("human", "{query}")
    ])

    def route_query(state: AgentState) -> AgentState:
        query = state["query"]
        response = assistant.llm.invoke(router_prompt.format(query=query))
        state["next_step"] = response.strip().lower()
        return state

    def execute_tool(state: AgentState) -> AgentState:
        tool_name = state["next_step"]
        query = state["query"]
        
        try:
            # Directly call the function from our mapping
            result = functions[tool_name](query)
            state["result"] = result
            state["messages"].append(AIMessage(content=result))
        except Exception as e:
            state["result"] = f"Error executing {tool_name}: {str(e)}"
            state["messages"].append(AIMessage(content=str(e)))
        
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
        "result": None
    }
    
    result = workflow.invoke(state)
    return result["result"] 