from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from db.database import engine, Base, get_db
from agent.langgraph_agent import app_agent
from langchain_core.messages import HumanMessage, AIMessage
import json

app = FastAPI(title="HCP CRM AI Assistant")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create database tables
@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)

class LogRequest(BaseModel):
    message: str

@app.post("/ai/log-interaction")
async def log_interaction(request: LogRequest):
    try:
        # Prepare state
        initial_state = {
            "messages": [HumanMessage(content=request.message)],
            "structured_data": {}
        }
        
        # Run LangGraph Agent
        result = await app_agent.ainvoke(initial_state)
        
        # Extract the most conversational AI message (last one with text that isn't just tool data)
        messages = result.get("messages", [])
        ai_message = "I've processed your request."
        
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                content = msg.content.strip()
                
                # If the message starts with JSON, we need to extract the text after it
                if content.startswith("{"):
                    import re
                    # Look for the last '}' and capture any natural text that follows it
                    # We look for characters that are NOT braces at the very end of the string
                    text_match = re.search(r'\}\s*([^{}]+)$', content, re.DOTALL)
                    if text_match:
                        cleaned = text_match.group(1).strip()
                        if cleaned:
                            ai_message = cleaned
                            break
                    
                    # If it's ONLY JSON, we go back in the history to find a conversational message
                    continue
                
                ai_message = content
                break
        
        return {
            "status": "success",
            "data": result.get("structured_data", {}),
            "response": ai_message
        }
    except Exception as e:
        import traceback
        import asyncio
        
        print("ERROR IN LOG_INTERACTION:")
        traceback.print_exc()
        
        error_msg = str(e)
        if isinstance(e, asyncio.CancelledError):
            error_msg = "AI Request timed out or was cancelled. Please try again."
        elif "GROQ_API_KEY" in error_msg or "401" in error_msg:
            error_msg = "API Key is missing or invalid. Check .env file."
        elif "429" in error_msg or "rate limit" in error_msg.lower():
            error_msg = "Groq API rate limit reached. Please wait a moment before trying again."
            
        raise HTTPException(status_code=500, detail=error_msg)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
