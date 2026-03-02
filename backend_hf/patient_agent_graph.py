import os
import logging
import operator
from typing import Annotated, TypedDict, List, Dict, Any, Optional, Union
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

# Configure logger
logger = logging.getLogger(__name__)

# --- Tools Definition ---

@tool
def search_medical_knowledge(query: str) -> str:
    """
    Search the medical knowledge base (RAG) for information about fractures, 
    diagnoses, treatments, and guidelines.
    Use this to answer questions where you need specific medical facts or context.
    """
    try:
        from backend_hf.app import KnowledgeAgent
    except ImportError:
        try:
             from app import KnowledgeAgent
        except ImportError:
             return "Error: Knowledge Agent not found."

    if not KnowledgeAgent:
        return "Error: Knowledge Agent is not available."
    
    try:
        agent = KnowledgeAgent()
        sources = agent.retrieve_sources(query)
        if not sources:
            # Fallback to definition lookup if query looks like a diagnosis
            summary = agent.get_medical_summary(query, 1.0)
            if "error" not in summary:
                return str(summary)
            return "No specific documents found."
        
        # Format sources
        formatted = "Found the following medical context:\n"
        for doc in sources:
            formatted += f"- [{doc.get('title', 'Doc')}]: {doc.get('content')}\n"
        return formatted
    except Exception as e:
        return f"Error retrieving knowledge: {e}"

@tool
def critique_diagnosis_logic(diagnosis: str, clinical_findings: str) -> str:
    """
    Ask the Critic Agent to review the logic of a diagnosis based on clinical findings.
    This acts as a safety check or second opinion.
    """
    # CriticAgent wraps Logic.
    # In the original code, CriticAgent.review_diagnosis needs an Image.
    # Here we mock the behavior for text-based critique or need to access the image from state.
    return f"Critic Agent Note: Please ensure '{diagnosis}' is consistent with findings: {clinical_findings}. (Full critique requires image)"


# --- Graph State ---

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    user_context: Dict[str, Any]
    medical_context: Dict[str, Any]
    inference_id: Optional[str]


# --- Nodes ---

def patient_interaction_agent(state: AgentState):
    """
    The main supervisor conversation node.
    """
    messages = state['messages']
    user_context = state.get('user_context', {})
    medical_context = state.get('medical_context', {})
    inference_id = state.get('inference_id')

    logger.info(f"Patient Agent Step. Inference ID in state: {inference_id}")
    
    # Construct System Prompt
    # Make inference_id explicitly available to the prompt context
    system_prompt = f"""You are MedAI's Patient Interaction Agent.
    You are an orchestrator that helps patients understand their fracture diagnosis.
    
    Current Patient Context:
    - Age: {user_context.get('age', 'N/A')}
    - History: {user_context.get('history', 'N/A')}
    
    Current Diagnosis Context:
    - Diagnosis: {medical_context.get('Diagnosis', 'N/A')}
    - Definition: {medical_context.get('Type_Definition', 'N/A')}
    
    Available Image ID: '{inference_id if inference_id else "NONE"}'
    
    You have access to the following specialized agents (tools):
    1. Knowledge Agent (search_medical_knowledge): For retrieving specific medical docs and guidelines.
    2. Critic Agent (critique_diagnosis_logic): For Validating logic/consistency.
    
    Decide whether to answer directly, or call a tool to get more information.
    
    Be empathetic, professional, but clarify you are an AI.
    """
    
    # Initialize Model
    api_key = os.environ.get("GEMINI_API_KEY")
    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")
    
    if not api_key:
        return {"messages": [AIMessage(content="System Error: API Key missing.")]}

    llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0.3)
    
    tools = [search_medical_knowledge, critique_diagnosis_logic]
    llm_with_tools = llm.bind_tools(tools)
    
    # Prepare messages
    # We must ensure SystemMessage is first
    sys_msg = SystemMessage(content=system_prompt)
    if not isinstance(messages[0], SystemMessage):
        all_messages = [sys_msg] + messages
    else:
        # Replace existing system message if present/stale
        all_messages = [sys_msg] + messages[1:]
        
    response = llm_with_tools.invoke(all_messages)
    
    return {"messages": [response]}


# --- Graph Construction ---

def create_patient_graph():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", patient_interaction_agent)
    
    # Add ToolNode
    tools = [search_medical_knowledge, critique_diagnosis_logic]
    tool_node = ToolNode(tools)
    workflow.add_node("tools", tool_node)
    
    # Set entry point
    workflow.add_edge(START, "agent")
    
    # Conditional edge
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
    )
    
    # From tools back to agent
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()

