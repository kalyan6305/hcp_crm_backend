from langchain.tools import tool
from pydantic import BaseModel, Field
from datetime import date, time
import json
from db.database import SessionLocal
from db.models import Interaction

class InteractionSchema(BaseModel):
    hcp_name: str = Field(description="Name of the Healthcare Professional")
    interaction_type: str = Field(description="Type: Meeting, Call, Email")
    date: str = Field(description="Date of interaction (YYYY-MM-DD)")
    time: str = Field(description="Time of interaction (HH:MM)")
    attendees: str = Field(description="List of attendees")
    topics_discussed: str = Field(description="Topics covered")
    materials_shared: str = Field(description="Sales materials provided")
    samples_distributed: str = Field(description="Medicine samples given")
    sentiment: str = Field(description="Positive, Neutral, or Negative")
    outcomes: str = Field(description="Results of the interaction")
    followup_actions: str = Field(description="Next steps required")
    ai_summary: str = Field(description="Brief summary of the meeting")

@tool(args_schema=InteractionSchema)
def log_interaction_tool(
    hcp_name: str,
    interaction_type: str = None,
    date: str = None,
    time: str = None,
    attendees: str = None,
    topics_discussed: str = None,
    materials_shared: str = None,
    samples_distributed: str = None,
    sentiment: str = None,
    outcomes: str = None,
    followup_actions: str = None,
    ai_summary: str = None
):
    """
    Extracts structured CRM fields and stores them in the database.
    """
    try:
        db = SessionLocal()
        
        # Safe date/time parsing
        int_date = None
        if date:
            try:
                int_date = date.fromisoformat(date)
            except:
                pass
        
        int_time = None
        if time:
            try:
                time_str = time[:5]
                int_time = time.fromisoformat(time_str)
            except:
                pass
        
        # Deduplication check: Check if same HCP, Date, and Time already exists
        existing = db.query(Interaction).filter(
            Interaction.hcp_name == (hcp_name or "Unknown"),
            Interaction.interaction_date == int_date,
            Interaction.interaction_time == int_time
        ).first()

        if existing:
            db.close()
            return f"Interaction for {hcp_name} on this date/time already exists in the database (ID: {existing.id}). No duplicate created."

        new_interaction = Interaction(
            hcp_name=hcp_name or "Unknown",
            interaction_type=interaction_type,
            interaction_date=int_date,
            interaction_time=int_time,
            attendees=attendees,
            topics_discussed=topics_discussed,
            materials_shared=materials_shared,
            samples_distributed=samples_distributed,
            sentiment=sentiment,
            outcomes=outcomes,
            followup_actions=followup_actions,
            ai_summary=ai_summary
        )
        
        db.add(new_interaction)
        db.commit()
        db.refresh(new_interaction)
        db.close()
        return f"Successfully logged interaction for {new_interaction.hcp_name} (ID: {new_interaction.id})"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error logging interaction: {str(e)}"

@tool
def edit_interaction_tool(interaction_id: int, update_data: str):
    """
    Modifies previously saved interaction data.
    """
    try:
        data = json.loads(update_data)
        db = SessionLocal()
        interaction = db.query(Interaction).filter(Interaction.id == interaction_id).first()
        
        if not interaction:
            db.close()
            return f"Interaction with ID {interaction_id} not found."
        
        for key, value in data.items():
            if hasattr(interaction, key):
                if key == "interaction_date" and value:
                    setattr(interaction, key, date.fromisoformat(value))
                elif key == "interaction_time" and value:
                    setattr(interaction, key, time.fromisoformat(value))
                else:
                    setattr(interaction, key, value)
        
        db.commit()
        db.close()
        return f"Successfully updated interaction {interaction_id}"
    except Exception as e:
        return f"Error updating interaction: {str(e)}"

@tool
def sentiment_detection_tool(notes: str):
    """
    Classifies HCP sentiment based on interaction notes as Positive, Neutral, or Negative.
    """
    # In a real scenario, this could be a specialized model, 
    # but for this agent, we'll let the LLM use logic or this tool description.
    # The agent will call this and we return a placeholder or simple logic.
    notes_lower = notes.lower()
    if any(word in notes_lower for word in ["happy", "positive", "interested", "great", "willing"]):
        return "Positive"
    if any(word in notes_lower for word in ["upset", "unhappy", "negative", "refused", "busy"]):
        return "Negative"
    return "Neutral"

@tool
def followup_suggestion_tool(outcomes: str, topics: str):
    """
    Recommends next actions like scheduling meeting, sending brochure, or clinical trial update.
    """
    suggestions = []
    if "interested" in outcomes.lower() or "positive" in outcomes.lower():
        suggestions.append("Schedule deep-dive clinical presentation")
    if "sample" in topics.lower():
        suggestions.append("Follow up on sample experience in 1 week")
    if not suggestions:
        suggestions.append("Send standard product brochure and monthly newsletter")
    
    return ", ".join(suggestions)

@tool
def material_recommendation_tool(topics_discussed: str):
    """
    Suggests relevant sales materials based on the discussion topics.
    """
    topics_lower = topics_discussed.lower()
    recommendations = []
    if "oncoboost" in topics_lower:
        recommendations.append("OncoBoost Phase 3 Trial Summary")
        recommendations.append("OncoBoost Safety Protocol Leaflet")
    if "efficacy" in topics_lower:
        recommendations.append("Comparative Efficacy Data (OncoBoost vs Competitors)")
    if not recommendations:
        recommendations.append("General Company HCP Portal Brochure")
    
    return ", ".join(recommendations)
