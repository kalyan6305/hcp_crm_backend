from typing import Annotated, TypedDict, List, Union
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from .groq_llm import get_groq_llm
from .gemini_llm import get_gemini_llm
from .tools import (
    log_interaction_tool, 
    edit_interaction_tool, 
    sentiment_detection_tool, 
    followup_suggestion_tool, 
    material_recommendation_tool
)
from pydantic import BaseModel
import json

# Define the tools
tools = [
    log_interaction_tool, 
    edit_interaction_tool, 
    sentiment_detection_tool, 
    followup_suggestion_tool, 
    material_recommendation_tool
]

# Define the state
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "The messages in the conversation"]
    structured_data: dict

def call_model(state: AgentState):
    # System prompt to guide extraction and handle off-topic chat
    system_prompt = (
        "You are an AI Medical CRM Assistant. Your goal is to help Sales Reps log interactions with HCPs. "
        "If the user provides interaction notes, use the tools to log them. "
        "IMPORTANT: You must respond ONLY with tool calls in standard JSON format when logging. Do not add any tags like <function=>. "
        "If the user is just greeting you or talking about unrelated topics, respond politely but DO NOT "
        "attempt to log any data or call tools. "
        "Each unique interaction should be logged only ONCE. If you have already successfully called "
        "log_interaction_tool for a specific meeting, do NOT call it again for that same meeting. "
        "CRITICAL: Do NOT echo your internal tool calls or JSON data in the chat. "
        "Your message must be 100% natural, conversational text ONLY. "
        "For example: 'I've successfully logged your meeting with Dr. Valli and updated the CRM record for you. Is there anything else you need?' "
        "Only extract: hcp_name, interaction_type, date, time, attendees, topics_discussed, "
        "materials_shared, samples_distributed, sentiment, outcomes, followup_actions, and ai_summary "
        "if they are explicitly or implicitly present in the notes."
    )

    try:
        llm = get_groq_llm()
        llm_with_tools = llm.bind_tools(tools)
        
        messages = state["messages"]
        if not any(isinstance(m, HumanMessage) and m.content.startswith("System:") for m in messages):
            messages = [HumanMessage(content=f"System: {system_prompt}")] + messages
            
        print("TRYING GROQ...")
        response = llm_with_tools.invoke(messages)
    except Exception as e:
        print(f"GROQ ERROR: {str(e)}. Falling back to Gemini...")
        llm = get_gemini_llm()
        llm_with_tools = llm.bind_tools(tools)
        
        messages = state["messages"]
        if not any(isinstance(m, HumanMessage) and m.content.startswith("System:") for m in messages):
            messages = [HumanMessage(content=f"System: {system_prompt}")] + messages
            
        response = llm_with_tools.invoke(messages)
        
    return {"messages": [response]}

def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

def extract_structured_data(state: AgentState):
    """
    Final node to ensure we have a JSON object for the frontend form.
    """
    try:
        llm = get_groq_llm()
    except Exception:
        llm = get_gemini_llm()
    
    messages = state["messages"]
    
    extraction_prompt = (
        "Extract HCP interaction details from the conversation above into a valid JSON object. "
        "Fields: hcp_name, interaction_type, date, time, attendees, topics_discussed, "
        "materials_shared, samples_distributed, sentiment, outcomes, followup_actions, ai_summary. "
        "CRITICAL: If the conversation is NOT about an HCP interaction (e.g., just a greeting like 'Hi' or 'Hello'), "
        "you MUST return an empty JSON object: {} "
        "Respond ONLY with the RAW JSON object."
    )
    
    extraction_msg = messages + [HumanMessage(content=extraction_prompt)]
    response = llm.invoke(extraction_msg)
    
    try:
        # Simple cleanup if LLM adds markdown backticks
        content = response.content.replace("```json", "").replace("```", "").strip()
        data = json.loads(content)
        return {"structured_data": data}
    except:
        return {"structured_data": {}}

# Create the graph
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("extract", extract_structured_data)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, {
    "tools": "tools",
    END: "extract"
})
workflow.add_edge("tools", "agent")
workflow.add_edge("extract", END)

# Compile
app_agent = workflow.compile()
