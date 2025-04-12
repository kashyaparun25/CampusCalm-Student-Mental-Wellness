import os
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# Import components from both modules
from updated_trag_ch import IntegratedMentalHealthBot, red_folder
from updated_vrag import MentalHealthVoiceAssistant, store_chat_history

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("campus_calm.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("campus_calm")

# Initialize app
app = FastAPI(title="Campus Calm Mental Health Assistant")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the bots
text_bot = IntegratedMentalHealthBot()
# The voice bot will be created per-request to ensure a fresh session

# Text chat endpoint
class TextRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

@app.post("/api/text-chat")
async def text_chat(request: TextRequest):
    """Process text-based chat messages"""
    try:
        logger.info(f"Text chat request: {request.message[:50]}...")
        
        # Process the text input
        response = text_bot.process_user_input(request.message)
        
        # Store in chat history if session_id is provided
        if request.session_id:
            store_chat_history(
                request.session_id, 
                request.message, 
                response, 
                text_bot.conversation_state["alert_flag"]
            )
        
        return {
            "success": True,
            "response": response,
            "alert_flag": text_bot.conversation_state["alert_flag"]
        }
    except Exception as e:
        logger.error(f"Error in text chat: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "response": "I'm having trouble processing your message. If this is urgent, please contact support directly."
        }

# Voice chat endpoint
class VoiceRequest(BaseModel):
    text: Optional[str] = None  # For text input fallback
    session_id: Optional[str] = None

@app.post("/api/voice-chat")
async def voice_chat(request: VoiceRequest = None):
    """Process voice-based chat interactions"""
    try:
        # Create a new voice assistant for each request 
        # (ensures clean state for speech recognition)
        voice_assistant = MentalHealthVoiceAssistant()
        
        # Use voice recognition if no text provided
        if not request or not request.text:
            user_input = voice_assistant.recognize_from_microphone()
            if not user_input:
                return {
                    "success": False,
                    "error": "No speech detected. Please speak clearly."
                }
        else:
            user_input = request.text
        
        # Process the input
        response = voice_assistant.process_user_input(user_input)
        
        # Convert to speech
        speech_success = voice_assistant.text_to_speech(response)
        
        # Use the session ID from the request if provided
        session_id = request.session_id if request and request.session_id else voice_assistant.session_id
        
        return {
            "success": True,
            "input": user_input,
            "response": response,
            "session_id": session_id,
            "speech_success": speech_success,
            "alert_flag": voice_assistant.conversation_state["alert_flag"]
        }
    except Exception as e:
        logger.error(f"Error in voice chat: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

# Get session history endpoint
@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Get chat history for a specific session"""
    try:
        from azure.cosmos import CosmosClient
        
        # Access environment variables
        cosmos_endpoint = os.getenv("COSMOSDB_ACCOUNT_URI")
        cosmos_key = os.getenv("COSMOSDB_ACCOUNT_KEY")
        database_name = os.getenv("HIST_DATABASE_NAME", "MentalHealthDb")
        container_name = os.getenv("HIST_CONTAINER_NAME", "chatHistory")
        
        # Connect to Cosmos DB
        client = CosmosClient(cosmos_endpoint, cosmos_key)
        database = client.get_database_client(database_name)
        container = database.get_container_client(container_name)
        
        # Query for this session's messages
        query = f"""
        SELECT * FROM c
        WHERE c.session_id = '{session_id}'
        ORDER BY c.timestamp
        """
        
        messages = list(container.query_items(
            query=query,
            enable_cross_partition_query=True
        ))
        
        return {
            "success": True,
            "session_id": session_id,
            "messages": messages
        }
    except Exception as e:
        logger.error(f"Error retrieving session: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

# List all sessions endpoint
@app.get("/api/sessions")
async def list_sessions():
    """List all available chat sessions"""
    try:
        from azure.cosmos import CosmosClient
        
        # Access environment variables
        cosmos_endpoint = os.getenv("COSMOSDB_ACCOUNT_URI")
        cosmos_key = os.getenv("COSMOSDB_ACCOUNT_KEY")
        database_name = os.getenv("HIST_DATABASE_NAME", "MentalHealthDb")
        container_name = os.getenv("HIST_CONTAINER_NAME", "chatHistory")
        
        # Connect to Cosmos DB
        client = CosmosClient(cosmos_endpoint, cosmos_key)
        database = client.get_database_client(database_name)
        container = database.get_container_client(container_name)
        
        # Query for unique session IDs with most recent timestamp
        query = """
        SELECT DISTINCT c.session_id, MAX(c.timestamp) as last_activity
        FROM c
        GROUP BY c.session_id
        ORDER BY MAX(c.timestamp) DESC
        """
        
        sessions = list(container.query_items(
            query=query,
            enable_cross_partition_query=True
        ))
        
        return {
            "success": True,
            "sessions": sessions
        }
    except Exception as e:
        logger.error(f"Error listing sessions: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "ok", "version": "1.0.0"}

# Interactive documentation route
@app.get("/")
async def root():
    """Redirect to API documentation"""
    return {"message": "Campus Calm API is running. Visit /docs for API documentation."}

# Run both components if executed directly
if __name__ == "__main__":
    import uvicorn
    
    # Print welcome message
    print("\n" + "="*60)
    print("🧠 Campus Calm Mental Health Assistant 🧠")
    print("="*60)
    
    # Run the FastAPI server
    print("\n💻 Starting the web server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)