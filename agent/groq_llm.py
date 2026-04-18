import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

def get_groq_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    
    model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    
    return ChatGroq(
        api_key=api_key,
        model=model_name,
        temperature=0.1
    )
