import os
import time
import json
import logging
import uuid
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
from azure.cosmos import CosmosClient
import openai
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AzureOpenAI

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all (change to localhost:3000 for strict setup)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mental_health_assistant.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("mental_health_assistant")

# Load environment variables
load_dotenv()

client = AzureOpenAI(
  api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
  api_version = "2024-02-01",
  azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") 
)

# Initialize History storage Cosmos DB 
COSMOS_ENDPOINT = os.getenv("COSMOSDB_ACCOUNT_URI")
COSMOS_KEY = os.getenv("COSMOSDB_ACCOUNT_KEY")
HIST_DATABASE_NAME = "MentalHealthDb"
HIST_CONTAINER_NAME = "chatHistory"

# Create the chat history container if it doesn't exist
try:
    cosmos_client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
    database = cosmos_client.create_database_if_not_exists(id=HIST_DATABASE_NAME)
    container = database.create_container_if_not_exists(
        id=HIST_CONTAINER_NAME,
        partition_key="/session_id",
        offer_throughput=400
    )
    logger.info(f"Chat history container initialized: {HIST_CONTAINER_NAME}")
except Exception as e:
    logger.error(f"Error initializing chat history container: {str(e)}")

def store_chat_history(session_id, user_input, bot_response, alert_flag=0):
    """Store chat exchange in Cosmos DB"""
    try:
        # Create the item to be stored
        item = {
            "id": str(uuid.uuid4()),
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "user_input": user_input,
            "bot_response": bot_response,
            "alert_flag": alert_flag
        }
        
        # Store in Cosmos DB
        try:
            # Access the container from the main module
            cosmos_client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
            database = cosmos_client.get_database_client(HIST_DATABASE_NAME)
            container = database.get_container_client(HIST_CONTAINER_NAME)
            
            # Create the item
            container.create_item(body=item)
            logger.info(f"Chat history stored in Cosmos DB with ID: {item['id']}")
            print(f"📝 Chat history saved to database (session: {session_id[:8]}...)")
                
        except Exception as db_error:
            logger.error(f"Failed to store chat history in Cosmos DB: {str(db_error)}")
            print(f"⚠️ Could not save chat history to database: {str(db_error)}")
    
    except Exception as e:
        logger.error(f"Error preparing chat history for storage: {str(e)}")
        print(f"⚠️ Error with chat history storage: {str(e)}")

class MentalHealthVoiceAssistant:
    def __init__(self):
        # System state with unique session ID
        self.session_id = str(uuid.uuid4())
        self.conversation_state = {
            "history": [],
            "is_greeting_phase": True,
            "alert_flag": 0  # Default alert flag is 0 (no alert)
        }
        self.logs = []
        
        # Azure Speech Service credentials
        self.speech_key = os.getenv("AZURE_SPEECH_KEY")
        self.speech_region = os.getenv("AZURE_SPEECH_REGION")
        
        logger.info(f"New session created: {self.session_id}")
        logger.info(f"Speech Key: {self.speech_key[:5]}... (first 5 chars only)")
        logger.info(f"Speech Region: {self.speech_region}")
        
        # Azure Cosmos DB NoSQL configuration
        self.cosmos_endpoint = os.getenv("COSMOSDB_ACCOUNT_URI")
        self.cosmos_key = os.getenv("COSMOSDB_ACCOUNT_KEY")
        self.cosmos_database = os.getenv("COSMOSDB_DATABASE")
        self.container_name = os.getenv("COSMOSDB_CONTAINER")
        
        # Azure OpenAI configuration
        self.openai_api_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        self.embeddings_deployment = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME")
        self.completions_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        self.model_name = os.getenv("AZURE_OPENAI_MODEL", "gpt-4o")
        
        # Campus Calm Agent IDs
        self.agent_id = os.environ.get("AGENT_ID", "asst_xblIbaWhZvDEFMFmGq6X7W9t")
        self.alert_agent_id = os.environ.get("ALERT_AGENT_ID", "asst_IolrPyItwouCxTrv9vwzAYPQ")
        
        # Email alert configuration
        self.smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.environ.get("SMTP_PORT", "587"))
        self.sender_email = os.environ.get("SENDER_EMAIL", "alert@mentalhealth.org")
        self.sender_password = os.environ.get("SENDER_PASSWORD", "")
        self.admin_email = os.environ.get("ADMIN_EMAIL", "admin@mentalhealth.org")
        
        # Print email configuration at startup
        print("📧 Email alert configuration:")
        print(f"  SMTP Server: {self.smtp_server}")
        print(f"  SMTP Port: {self.smtp_port}")
        print(f"  Sender Email: {self.sender_email}")
        print(f"  Admin Email: {self.admin_email}")
        print(f"  Sender Password Set: {'Yes' if self.sender_password else 'No - Authentication will fail'}")
        
        # Console output settings
        self.verbose_console = True
        
        # Initialize services
        self.initialize_services()
    
    def initialize_services(self):
        """Initialize all required services"""
        self.initialize_speech_service()
        self.initialize_cosmos_db()
        self.initialize_openai()
        self.initialize_campus_calm_agent()

    def test_email_alert(self):
        """Test the email alert functionality"""
        print("\n🧪 Testing email alert functionality...")
        test_reason = "TEST ALERT - Please ignore this test message"
        self.trigger_alert(test_reason)
        return True
        
    def initialize_speech_service(self):
        """Initialize Azure Speech services"""
        try:
            self.speech_config = speechsdk.SpeechConfig(
                subscription=self.speech_key, 
                region=self.speech_region
            )
            
            # Set speech recognition and synthesis language
            self.speech_config.speech_recognition_language = "en-US"
            self.speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"
            logger.info("Speech config initialized successfully")
            
        except Exception as e:
            error_msg = f"Error initializing speech config: {str(e)}"
            logger.error(error_msg)
            # We'll continue even if speech fails, as we can fall back to text mode
    
    def initialize_cosmos_db(self):
        """Initialize Cosmos DB client for RAG knowledge base"""
        try:
            self.cosmos_client = CosmosClient(self.cosmos_endpoint, self.cosmos_key)
            self.database = self.cosmos_client.get_database_client(self.cosmos_database)
            self.container = self.database.get_container_client(self.container_name)
            logger.info(f"Connected to Cosmos DB container: {self.container_name}")
            
            # Validate container has documents
            doc_count = self.count_documents()
            logger.info(f"Container has {doc_count} documents")
            print(f"📁 Connected to Cosmos DB - {doc_count} documents available")
            
            if doc_count == 0:
                logger.warning("No documents found in container. The RAG system may not work properly.")
                print("⚠️ Warning: No documents found in the knowledge base")
                
        except Exception as e:
            error_msg = f"Error initializing Cosmos DB: {str(e)}"
            logger.error(error_msg)
            print(f"❌ Cosmos DB Error: {str(e)}")
            # Explicitly set attributes to None to prevent attribute errors
            self.cosmos_client = None
            self.database = None
            self.container = None
    
    def initialize_openai(self):
        """Initialize Azure OpenAI configuration"""
        try:
            client = AzureOpenAI(
                api_key = os.getenv("OPENAI_API_KEY"),  
                api_version = "2024-02-01",
                azure_endpoint =os.getenv("AZURE_OPENAI_ENDPOINT") 
                )
            logger.info("OpenAI configuration initialized")
            
        except Exception as e:
            error_msg = f"Error initializing OpenAI: {str(e)}"
            logger.error(error_msg)
            raise
    
    def initialize_campus_calm_agent(self):
        """Initialize Campus Calm agent for mental health assessment"""
        try:
            # Import Azure libraries for Campus Calm agent
            try:
                from azure.ai.projects import AIProjectClient
                from azure.identity import DefaultAzureCredential
                
                azure_conn_string = os.environ.get("AZURE_CONN_STRING", 
                    "eastus2.api.azureml.ms;f763e218-bbdb-4330-95b6-f36e504b0440;rg-rtiwari3-8465_ai;campus_calm_ver1")
                
                self.ai_client = AIProjectClient.from_connection_string(
                    credential=DefaultAzureCredential(),
                    conn_str=azure_conn_string
                )
                logger.info("Azure AI Project client initialized successfully")
                
            except ImportError:
                error_msg = "Azure AI Projects libraries not found. Mental health assessment will be limited."
                logger.warning(error_msg)
                self.ai_client = None
                
        except Exception as e:
            error_msg = f"Failed to initialize Campus Calm agent: {str(e)}"
            logger.error(error_msg)
            self.ai_client = None
    
    def count_documents(self) -> int:
        """Count documents in the Cosmos DB container"""
        try:
            if not hasattr(self, 'container') or self.container is None:
                return 0
                
            query = "SELECT VALUE COUNT(1) FROM c"
            count_results = list(self.container.query_items(
                query=query,
                enable_cross_partition_query=True
            ))
            return count_results[0] if count_results else 0
        except Exception as e:
            logger.error(f"Error counting documents: {e}")
            return 0
    
    def generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for a text using Azure OpenAI."""
        try:
            logger.info(f"Generating embeddings using deployment: {self.embeddings_deployment}")
            
            response = client.embeddings.create(
                input=text,
                model=self.embeddings_deployment  # Use model parameter
            )
            
            embeddings = response.data[0].embedding  # Changed from dictionary access to object properties
            logger.info(f"Successfully generated embeddings with length: {len(embeddings)}")
            
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return []
    
    def recognize_from_microphone(self) -> str:
        """
        Recognize speech from microphone input
        Returns the recognized text
        """
        try:
            # Create a speech recognizer and start recognition
            audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
            speech_recognizer = speechsdk.SpeechRecognizer(
                speech_config=self.speech_config, 
                audio_config=audio_config
            )
            
            logger.info("Listening for speech input...")
            
            # Start speech recognition
            result = speech_recognizer.recognize_once_async().get()
            
            # Process the result
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                logger.info(f"Recognized: {result.text}")
                return result.text
            elif result.reason == speechsdk.ResultReason.NoMatch:
                logger.warning("No speech could be recognized")
                return ""
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation = result.cancellation_details
                logger.error(f"Speech Recognition canceled: {cancellation.reason}")
                if cancellation.reason == speechsdk.CancellationReason.Error:
                    logger.error(f"Error details: {cancellation.error_details}")
                return ""
        except Exception as e:
            logger.error(f"Error in speech recognition: {str(e)}")
            return ""
    
    def text_to_speech(self, text: str) -> bool:
        """
        Convert text to speech and play it
        Returns success status
        """
        try:
            # Create a speech synthesizer
            speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config)
            
            # Synthesize the text
            result = speech_synthesizer.speak_text_async(text).get()
            
            # Check result
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                logger.info("Text-to-speech completed successfully")
                return True
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation = result.cancellation_details
                logger.error(f"Speech synthesis canceled: {cancellation.reason}")
                if cancellation.reason == speechsdk.CancellationReason.Error:
                    logger.error(f"Error details: {cancellation.error_details}")
                return False
            
            return False
        except Exception as e:
            logger.error(f"Error in text-to-speech: {str(e)}")
            return False
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from the text for search."""
        words = text.lower().split()
        stopwords = {"a", "an", "the", "and", "or", "but", "if", "then", "else", "when", 
                    "at", "from", "by", "for", "with", "about", "against", "between", 
                    "into", "through", "during", "before", "after", "above", "below", 
                    "to", "of", "in", "on", "is", "are", "was", "were", "be", "been", 
                    "have", "has", "had", "do", "does", "did", "should", "could", "would"}
        keywords = [word for word in words if word not in stopwords and len(word) > 3]
        return list(set(keywords))
    
    def vector_search(self, query: str, top_k: int = 3) -> List[dict]:
        """Perform a vector search using query embeddings with proper Cosmos DB syntax."""
        try:
            start_time = time.time()
            print(f"🔍 Performing vector search for: '{query}'")
            
            # Check if container exists first
            if not hasattr(self, 'container') or self.container is None:
                logger.warning("No Cosmos DB container available. Returning empty results.")
                print("⚠️ Cosmos DB container not available. Cannot perform vector search.")
                return []
                
            # Generate embeddings for the query
            query_embedding = self.generate_embeddings(query)
            
            if not query_embedding:
                logger.warning("Could not generate embeddings for query, falling back to basic search")
                print("⚠️ Could not generate embeddings, falling back to keyword search")
                return self.basic_search(query, top_k)
            
            if not query_embedding:
                logger.warning("Could not generate embeddings for query, falling back to basic search")
                return self.basic_search(query, top_k)
            
            # Execute the vector search with the correct syntax
            logger.info(f"Executing vector search with embedding length: {len(query_embedding)}")
            
            # Use the documented VectorDistance syntax with all parameters
            options_json = "{'distanceFunction':'cosine','dataType':'float32'}"
            
            query_str = f"""
            SELECT TOP {top_k} c.id, c.content, c.metadata,
            VectorDistance(c.embedding, @queryEmbedding, false, {options_json}) AS similarity
            FROM c 
            WHERE IS_ARRAY(c.embedding)
            ORDER BY VectorDistance(c.embedding, @queryEmbedding, false, {options_json})
            """
            
            results = list(self.container.query_items(
                query=query_str,
                parameters=[{"name": "@queryEmbedding", "value": query_embedding}],
                enable_cross_partition_query=True
            ))
            
            logger.info(f"Vector search returned {len(results)} results")
            
            return results
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            logger.info("Falling back to basic search...")
            return self.basic_search(query, top_k)
    
    def basic_search(self, query: str, top_k: int = 3) -> List[dict]:
        """Perform a basic keyword search on the content stored in Cosmos DB."""
        try:

            if not hasattr(self, 'container') or self.container is None:
                logger.warning("No Cosmos DB container available. Returning empty results.")
                print("⚠️ Cosmos DB container not available. Cannot perform keyword search.")
                return []
            query_keywords = self.extract_keywords(query)
            
            if not query_keywords:
                logger.warning("No meaningful keywords extracted from the query")
                query_text = f"SELECT TOP {top_k} c.content, c.metadata FROM c ORDER BY c._ts DESC"
                items = list(self.container.query_items(
                    query=query_text,
                    enable_cross_partition_query=True
                ))
                return items
            
            keyword_conditions = []
            for keyword in query_keywords:
                if len(keyword) > 3:  # Only use keywords longer than 3 characters
                    keyword_conditions.append(f"CONTAINS(c.content, '{keyword}')")
            
            if not keyword_conditions:
                query_text = f"SELECT TOP {top_k} c.content, c.metadata FROM c ORDER BY c._ts DESC"
            else:
                sql_condition = " OR ".join(keyword_conditions)
                query_text = f"SELECT TOP {top_k} c.content, c.metadata FROM c WHERE {sql_condition}"
            
            items = list(self.container.query_items(
                query=query_text,
                enable_cross_partition_query=True
            ))
            return items
        except Exception as e:
            logger.error(f"Error performing basic search: {e}")
            return []
    
    def is_greeting(self, text: str) -> bool:
        """Check if the message is a simple greeting"""
        greetings = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"]
        return any(greeting in text.lower() for greeting in greetings) and len(text.split()) < 5
    
    def is_expressing_concern(self, text: str) -> bool:
        """Check if user is likely expressing an emotional concern"""
        concern_indicators = ["feel", "feeling", "sad", "anxious", "stress", "depress", "worry", 
                             "overwhelm", "tired", "exhaust", "lonely", "afraid", "fear", "upset", 
                             "struggle", "difficult", "hard time", "low", "down"]
        return any(indicator in text.lower() for indicator in concern_indicators)
    
    def is_farewell(self, text: str) -> bool:
        """Check if the user is saying goodbye"""
        farewells = ["exit", "bye", "goodbye", "see you", "farewell", "Bye Loona", "See you Loona"]
        return any(farewell in text.lower() for farewell in farewells)
    
    def analyze_mental_health_severity(self, user_input: str) -> Dict[str, Any]:
        """Analyze user input to determine if severe mental health issues are present"""
        if self.ai_client is None:
            # If Campus Calm agent isn't available, use OpenAI fallback
            return self.analyze_mental_health_severity_openai(user_input)
            
        # Build context from conversation history
        all_history = ""
        if len(self.conversation_state["history"]) > 0:
            all_history = "\n".join([f"{item['role'].upper()}: {item['content']}" for item in self.conversation_state["history"]])
        
        # Create a prompt for risk assessment
        prompt = f"""
        COMPLETE CONVERSATION HISTORY:
        {all_history}
        
        CURRENT USER MESSAGE: {user_input}
        
        As a mental health professional, analyze the conversation history and the current message.
        Determine if there are signs of severe depression, anxiety, potential self-harm, or other
        serious mental health concerns that would require urgent attention.
        
        Return a JSON object in this format:
        {{
            "alert_flag": 0 or 1,
            "reason": "explanation if alert_flag is 1, otherwise empty"
        }}
        
        Set alert_flag to 1 ONLY if you detect clear indications of:
        - you have to trigger for the key words: "suicidal", "self-harm", "hopelessness", "severe anxiety", "crisis", Kill, "die", "no reason to live", "Harm others"
        - Suicidal thoughts or intentions
        - Self-harm thoughts or behaviors
        - Severe depression with hopelessness
        - Severe anxiety that is debilitating
        - Crisis situation that requires immediate intervention
        - Statements about harming self or having no reason to live
        - Explicit mention of a suicide plan or method
        
        Make a careful and reasoned assessment. Only set alert_flag to 1 for genuinely concerning situations.
        """
        
        try:
            # Create a new thread for this interaction
            thread = self.ai_client.agents.create_thread()
            
            # Send message to the agent
            self.ai_client.agents.create_message(
                thread_id=thread.id,
                role="user",
                content=prompt
            )
            
            # Process the message
            run = self.ai_client.agents.create_and_process_run(
                thread_id=thread.id,
                agent_id=self.agent_id
            )
            
            # Get the response
            messages = self.ai_client.agents.list_messages(thread_id=thread.id)
            
            # Find the assistant message
            response_text = ""
            for message in messages.data:
                if message.role == "assistant":
                    for content_item in message.content:
                        if hasattr(content_item, 'text') and hasattr(content_item.text, 'value'):
                            response_text = content_item.text.value
                            break
                    break
            
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'({.*})', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                result = json.loads(json_str)
            else:
                # If not found, try parsing the entire response
                result = json.loads(response_text)
            
            # Ensure the result has the expected format
            if "alert_flag" not in result:
                logger.warning(f"Invalid severity analysis response format: {response_text}")
                return {"alert_flag": 0, "reason": ""}
            
            # Log the analysis result
            alert_status = "🚨 ALERT FLAG RAISED" if result.get("alert_flag") == 1 else "✅ No serious concerns detected"
            logger.info(f"Mental health analysis result: {alert_status}")
            if result.get("alert_flag") == 1 and result.get("reason"):
                logger.info(f"Alert reason: {result.get('reason')}")
                
            return result
        except Exception as e:
            logger.error(f"Error in mental health severity analysis with Campus Calm agent: {str(e)}")
            # Fall back to OpenAI analysis
            return self.analyze_mental_health_severity_openai(user_input)
    
    def analyze_mental_health_severity_openai(self, user_input: str) -> Dict[str, Any]:
        """Analyze mental health severity using Azure OpenAI as a fallback"""
        try:
            # Build context from conversation history
            all_history = ""
            if len(self.conversation_state["history"]) > 0:
                all_history = "\n".join([f"{item['role'].upper()}: {item['content']}" for item in self.conversation_state["history"]])
            
            # Create system prompt for analysis
            system_prompt = f"""
            You are a mental health professional tasked with assessing the severity of a user's mental health concerns.
            Analyze the conversation history and current message to determine if there are signs of severe issues.
            
            You must respond ONLY with a valid JSON object in this exact format:
            {{
                "alert_flag": 0 or 1,
                "reason": "explanation if alert_flag is 1, otherwise empty"
            }}
            
            Set alert_flag to 1 ONLY if you detect clear indications of:
            - you have to trigger for the key words: "suicidal", "self-harm", "hopelessness", "severe anxiety", "crisis", Kill, "die", "no reason to live", "Harm others"
            - Suicidal thoughts or intentions
            - Self-harm thoughts or behaviors
            - Severe depression with hopelessness
            - Severe anxiety that is debilitating
            - Crisis situation that requires immediate intervention
            - Statements about harming self or having no reason to live
            - Explicit mention of a suicide plan or method
            
            Make a careful assessment. Only set alert_flag to 1 for genuinely concerning situations.
            """
            
            # Format user content with history
            user_content = f"""
            COMPLETE CONVERSATION HISTORY:
            {all_history}
            
            CURRENT USER MESSAGE: {user_input}
            
            Analyze for mental health severity and respond ONLY with the required JSON format.
            """
            
            # Make API call
            response = client.chat.completions.create(
                model=self.completions_deployment,
                #deployment_id=COMPLETIONS_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ]
                  # Use low temperature for deterministic response
            )
            
            response_text = response.choices[0].message.content
            
            # Parse the JSON response
            try:
                result = json.loads(response_text)
                
                # Validate the result format
                if "alert_flag" not in result:
                    logger.warning(f"Invalid severity analysis response format: {response_text}")
                    return {"alert_flag": 0, "reason": ""}
                
                # Log the analysis result
                alert_status = "🚨 ALERT FLAG RAISED" if result.get("alert_flag") == 1 else "✅ No serious concerns detected"
                logger.info(f"OpenAI Mental health analysis result: {alert_status}")
                if result.get("alert_flag") == 1 and result.get("reason"):
                    logger.info(f"Alert reason: {result.get('reason')}")
                
                return result
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse severity analysis as JSON: {response_text}")
                return {"alert_flag": 0, "reason": ""}
                
        except Exception as e:
            logger.error(f"Error in mental health severity analysis with OpenAI: {str(e)}")
            # Return safe default if all else fails
            return {"alert_flag": 0, "reason": ""}
    
    def trigger_alert(self, reason: str) -> None:
        """Trigger the alert process for concerning mental health situations"""
        logger.warning(f"MENTAL HEALTH ALERT TRIGGERED: {reason}")
        print(f"\n🚨 MENTAL HEALTH ALERT TRIGGERED: {reason}")
        
        # Build context from entire conversation history
        all_history = ""
        if len(self.conversation_state["history"]) > 0:
            all_history = "\n".join([f"{item['role'].upper()}: {item['content']}" for item in self.conversation_state["history"]])
        
        # Print email settings for debugging
        print("\n📧 Email Alert Settings:")
        print(f"  SMTP Server: {self.smtp_server}")
        print(f"  SMTP Port: {self.smtp_port}")
        print(f"  Sender Email: {self.sender_email}")
        print(f"  Admin Email: {self.admin_email}")
        print(f"  Sender Password Set: {'Yes' if self.sender_password else 'No - Authentication may fail'}")
        
        # Try to use the Campus Calm alert agent if available
        if self.ai_client is not None:
            try:
                print("📧 Generating alert email with Campus Calm agent...")
                
                # Create a prompt for the alert agent
                prompt = f"""
                COMPLETE CONVERSATION HISTORY:
                {all_history}
                
                ALERT REASON: {reason}
                
                You are an alert system for a mental health chatbot. A potential mental health crisis has been detected.
                Draft a concise email to the administrator that includes:
                
                1. A clear subject line indicating this is an urgent mental health alert
                2. A brief summary of the concerning aspects of the conversation
                3. The specific trigger that caused the alert: {reason}
                4. A condensed version of the relevant parts of the conversation
                5. Any recommendations for follow-up actions
                
                Format the response as a JSON object with these fields:
                {{
                    "subject": "Email subject line",
                    "recipient": "admin@organization.com",
                    "body": "Complete email body with all required information"
                }}
                """
                
                # Create a new thread for this interaction
                thread = self.ai_client.agents.create_thread()
                
                # Send message to the alert agent
                self.ai_client.agents.create_message(
                    thread_id=thread.id,
                    role="user",
                    content=prompt
                )
                
                # Process the message
                run = self.ai_client.agents.create_and_process_run(
                    thread_id=thread.id,
                    agent_id=self.alert_agent_id
                )
                
                # Get the response
                messages = self.ai_client.agents.list_messages(thread_id=thread.id)
                
                # Find the assistant message
                response_text = ""
                for message in messages.data:
                    if message.role == "assistant":
                        for content_item in message.content:
                            if hasattr(content_item, 'text') and hasattr(content_item.text, 'value'):
                                response_text = content_item.text.value
                                break
                        break
                
# Try to extract JSON from the response
                import re
                json_match = re.search(r'({.*})', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    email_data = json.loads(json_str)
                else:
                    # If not found, try parsing the entire response
                    email_data = json.loads(response_text)
                
                # Print email details for debugging
                print("\n📧 Alert Email Generated:")
                print(f"  Subject: {email_data.get('subject', 'URGENT: Mental Health Alert')}")
                print(f"  Recipient: {email_data.get('recipient', self.admin_email)}")
                print("  Body:")
                print("  " + "\n  ".join(email_data.get('body', '').split('\n')[:5]) + "...")
                
                # Send the email
                success = self.send_alert_email(email_data)
                if not success:
                    print("⚠️ Failed to send email with Campus Calm agent. Trying fallback...")
                    # Try fallback if primary method fails
                    self.send_fallback_alert_email(reason, all_history)
                
            except Exception as e:
                logger.error(f"Error using Campus Calm alert agent: {str(e)}")
                print(f"⚠️ Error generating alert email: {str(e)}. Using fallback alert.")
                # Fall back to simpler alert email
                self.send_fallback_alert_email(reason, all_history)
        else:
            # Use fallback alert email if Campus Calm agent isn't available
            print("📧 Campus Calm agent not available. Generating fallback alert email...")
            self.send_fallback_alert_email(reason, all_history)
    
    def send_alert_email(self, email_data: Dict[str, str]) -> bool:
        """Send an alert email using SMTP with simple HTML formatting"""
        try:
            # Default recipient if not specified
            recipient = email_data.get("recipient", self.admin_email)
            recipients_list = [recipient]  # Create a list as required by sendmail
            
            # Create message
            msg = MIMEMultipart()
            msg["From"] = self.sender_email
            msg["To"] = recipient
            msg["Subject"] = email_data.get("subject", "URGENT: Mental Health Alert")
            
            # Format the body with HTML for better readability, but with minimal styling
            body_text = email_data.get("body", "A potential mental health crisis has been detected. Please review the conversation.")
            reason = email_data.get("reason", "Mental health concern detected")
            
            # Simplified HTML body without complex CSS styling
            html_body = """
            <html>
            <body>
                <div style="background-color: #f44336; color: white; padding: 10px; text-align: center;">
                    <h2>⚠️ MENTAL HEALTH ALERT - IMMEDIATE ATTENTION REQUIRED ⚠️</h2>
                </div>
                <div style="padding: 15px;">
                    <p><strong>Alert Reason:</strong> {0}</p>
                    
                    <p style="color: #666; font-size: 0.9em;"><strong>Timestamp:</strong> {1}</p>
                    <p style="color: #666; font-size: 0.9em;"><strong>Session ID:</strong> {2}</p>
                    
                    <h3>Conversation History:</h3>
                    <div style="background-color: #f9f9f9; padding: 10px; border-left: 4px solid #ccc; margin: 10px 0;">
                        {3}
                    </div>
                    
                    <p><strong>This is an automated alert from the Mental Health Bot system.</strong><br>
                    Please review the conversation immediately and take appropriate action.</p>
                    
                    <div style="font-size: 0.8em; color: #666; margin-top: 20px;">
                        <p>Campus Calm Mental Health Alert System<br>
                        This email was sent automatically in response to concerning user input.</p>
                    </div>
                </div>
            </body>
            </html>
            """.format(
                reason,
                datetime.now().isoformat(),
                self.session_id,
                body_text.replace("Conversation History:", "").replace("--------------------", "").replace("\n", "<br>")
            )
            
            # Also create a plain text alternative
            plain_text = f"""
    MENTAL HEALTH ALERT - IMMEDIATE ATTENTION REQUIRED

    Alert Reason: {reason}

    Timestamp: {datetime.now().isoformat()}
    Session ID: {self.session_id}

    Conversation History:
    --------------------
    {body_text}

    This is an automated alert from the Mental Health Bot system.
    Please review the conversation immediately and take appropriate action.
            """
            
            # Create a MIME multipart/alternative message
            msg_alt = MIMEMultipart('alternative')
            
            # Attach plain text and HTML versions
            msg_alt.attach(MIMEText(plain_text, 'plain'))
            msg_alt.attach(MIMEText(html_body, 'html'))
            
            # Attach the alternative part to the main message
            msg.attach(msg_alt)
            
            print(f"\n📧 Attempting to send email via {self.smtp_server}:{self.smtp_port}...")
            
            # Create SMTP connection with detailed logging
            try:
                s = smtplib.SMTP(self.smtp_server, self.smtp_port)
                s.set_debuglevel(1)  # Enable verbose debug output
                print("  SMTP connection established")
                
                s.starttls()
                print("  TLS started successfully")
                
                s.login(self.sender_email, self.sender_password)
                print("  SMTP authentication successful")
                
                # Send the email - use the proper method with a list of recipients
                msg_text = msg.as_string()
                send_errors = s.sendmail(self.sender_email, recipients_list, msg_text)
                
                # Check for any send errors
                if len(send_errors) == 0:
                    print("  Message sent successfully")
                else:
                    print(f"  ⚠️ Partial send errors: {send_errors}")
                    
                # Close the connection properly
                s.quit()
                
                logger.info(f"Alert email sent to {recipient}")
                print(f"✅ Alert email sent to {recipient}")
                return True
                    
            except Exception as e:
                print(f"  ⚠️ SMTP Error: {str(e)}")
                raise
            
        except Exception as e:
            logger.error(f"Failed to send email: {str(e)}")
            print(f"❌ Email Error: {str(e)}")
            
            # Print email details that would have been sent
            print("\n📧 Email that would have been sent:")
            print(f"  To: {email_data.get('recipient', self.admin_email)}")
            print(f"  Subject: {email_data.get('subject', 'URGENT: Mental Health Alert')}")
            print(f"  Body Preview: " + body_text[:100] + "...")
            
            return False

    def send_fallback_alert_email(self, reason: str, conversation_history: str) -> bool:
        """Send a basic alert email with minimal formatting when the agent isn't available"""
        try:
            subject = "URGENT: Mental Health Alert - Immediate Attention Required"
            
            # Create a more formatted body with linebreaks to make it more readable
            body = f"""
            MENTAL HEALTH ALERT - IMMEDIATE ATTENTION REQUIRED
            
            Alert Reason: {reason}
            
            Timestamp: {datetime.now().isoformat()}
            Session ID: {self.session_id}
            
            Conversation History:
            --------------------
            {conversation_history}
            
            This is an automated alert from the Mental Health Bot system.
            Please review the conversation immediately and take appropriate action.
            """
            
            email_data = {
                "subject": subject,
                "recipient": self.admin_email,
                "body": body,
                "reason": reason  # Add reason explicitly for HTML formatting
            }
            
            # Print email details for debugging
            print("\n📧 Fallback Alert Email Generated:")
            print(f"  Subject: {subject}")
            print(f"  Recipient: {self.admin_email}")
            print("  Body Preview:")
            print("  " + "\n  ".join(body[:500].split('\n')) + "...")
            
            return self.send_alert_email(email_data)
                
        except Exception as e:
            logger.error(f"Failed to send fallback alert email: {str(e)}")
            print(f"❌ Fallback Email Error: {str(e)}")
            return False
    
    def generate_response_with_rag(self, query: str, mental_health_analysis: Dict[str, Any] = None) -> str:
        """Generate a response using RAG approach with mental health considerations"""
        try:
            # Get relevant documents from knowledge base
            try:
                results = self.vector_search(query)
                if not results:
                    logger.warning("No vector search results, falling back to basic search")
                    results = self.basic_search(query)
            except Exception as e:
                logger.error(f"Search failed: {e}")
                logger.warning("Falling back to basic search")
                results = self.basic_search(query)
            
            # Format the context from documents
            context = "\n\n".join([doc.get("content", "") for doc in results])
            
            # Get mental health alert status
            is_alert = (mental_health_analysis is not None and 
                        mental_health_analysis.get("alert_flag", 0) == 1)
            
            # Initialize system_message with default value first
            system_message = ""
            
            # Create appropriate system message based on mental health analysis
            if is_alert:
                # High concern system message
                system_message = f"""
                You are Loona, a Gen Z mental health assistant created specifically for Stevens Institute of Technology students. Your vibe is supportive but casual - like texting with a wise friend who's been through it all.

                Always begin each new conversation with a friendly, Gen Z-style greeting such as:
                "Hey there! I'm Loona, your mental health bestie here at Stevens. What's on your mind today?"
                or
                "Sup! Loona here. I'm your go-to for mental health support at Stevens. How's it going?"

                Context information (Knowledge Base):
                {context}

                {red_folder}

                Core Instructions & Guidelines:
                Talk in small bits, keep it short and sweet. Keep it like a conversation between friends. Do not use emojis. Do not load the user with information. Keep it simple and easy to understand.
                Dont pity the user. Be supportive and encouraging.
                0. You are the primary point of contact you are responsible to chat with user, empathize and advice. Only if you are not able to help, you can refer to the resources below.
                1. To answer the user's question, you can use the information from the red folder stored in the RAG container database. You can also use the information from the knowledge base.
                1. Source Limitation: Always prioritize information from the Stevens Red Folder stored in the RAG container database.
                2. Crisis Detection: If user shows signs of mental health crisis, immediately provide CAPS 24/7 crisis line: (201) 216-5177.
                3. CAPS Information: Counseling & Psychological Services (CAPS)
                    Student Wellness Center, 2nd Floor
                    201-216-5177
                    caps@stevens.edu
                    Phone line is staffed 24/7
                    Visit stevens.edu/caps for more information
                4. UWill Details: Include UWill as 24/7 teletherapy option (free, confidential, uwill.com/stevens).
                5. Tone: Use Gen Z lingo (e.g., "vibe check," "no cap," "lowkey," "bestie," "fr") and keep it conversational.
                6. Conciseness: Keep responses short (1-3 sentences when possible) and easy to understand.
                7. Response Structure: Address concern → provide Stevens resources → add encouragement → optional follow-up question.
                8. Style: Sound like a supportive peer rather than a clinical professional, do NOT use emoji.
                8.1. Avoid using phrases like "It's important to" or "It's crucial to". Instead, use phrases like "You should" or "You might want to". For example, instead of "It's important to talk to someone about this", say "You should talk to someone about this".
                9. Resource Priority: Stevens CAPS first, UWill second, external resources (988, text HOME to 741741) last.
                10. Special Cases: For academic stress mention CAPS workshops; for relationship issues highlight counseling options.
                11. Follow-up: Always end with a supportive question or statement.
                12. Confidentiality: Assure users of confidentiality and privacy.
                13. Escalation: If user expresses suicidal thoughts, immediately provide CAPS 24/7 crisis line: (201) 216-5177.
                

                Remember: You're a friendly guide to Stevens mental health resources, not a therapist. Your goal is to connect students with the right campus support systems while being relatable and approachable.
                """

                # For crisis situations
                if mental_health_analysis.get("alert_flag") == 1:
                    system_message = f"""
                    You are Loona, a Gen Z mental health assistant for Stevens Institute of Technology students.
                    
                    IMPORTANT: The alert flag has been triggered for the following reason:
                    {mental_health_analysis.get("reason", "Potential mental health crisis detected")}
                    
                    Context information (Knowledge Base):
                    {context}

                    {red_folder}
                    
                    Your primary goals are to:
                    Dont pity the user. Be supportive and encouraging.
                    Talk in small bits, keep it short and sweet. Keep it like a conversation between friends. Do not use emojis. Do not load the user with information. Keep it simple and easy to understand.
                    0. You are the primary point of contact you are responsible to chat with user, empathize and advice. Only if you are not able to help, you can refer to the resources below.
                    1. To answer the user's question, you can use the information from the red folder stored in the RAG container database. You can also use the information from the knowledge base.
                    2. Immediately provide CAPS 24/7 crisis line: (201) 216-5177
                    3. Mention ProtoCall Services for after-hours support
                    4. Include National Crisis resources: 988 or text HOME to 741741
                    5. Keep your response clear, supportive, and super concise
                    6. Use a tone that's serious but still approachable
                    
                    Do not use excessive emojis or slang in crisis situations. Focus on connecting the student with immediate professional help from CAPS or crisis services.
                    """
            else:
                # Add a default system message for non-alert situations
                system_message = f"""
                You are Loona, a Gen Z mental health assistant created specifically for Stevens Institute of Technology students. Your vibe is supportive but casual - like texting with a wise friend who's been through it all.

                Context information (Knowledge Base):
                {context}

                Core Instructions & Guidelines:
                Dont pity the user. Be supportive and encouraging.
                Talk in small bits, keep it short and sweet. Keep it like a conversation between friends. Do not use emojis. Do not load the user with information. Keep it simple and easy to understand.
                0. You are the primary point of contact you are responsible to chat with user, empathize and advice. Only if you are not able to help, you can refer to the resources below.
                1. To answer the user's question, you can use the information from the red folder stored in the RAG container database. You can also use the information from the knowledge base.
                1. Source Limitation: Always prioritize information from the Stevens Red Folder stored in the RAG container database.
                2.. CAPS Information: Counseling & Psychological Services (CAPS)
                    Student Wellness Center, 2nd Floor
                    201-216-5177
                    caps@stevens.edu
                    Phone line is staffed 24/7
                    Visit stevens.edu/caps for more information
                3. UWill Details: Include UWill as 24/7 teletherapy option (free, confidential, uwill.com/stevens).
                4. Tone: Use Gen Z lingo (e.g., "vibe check," "no cap," "lowkey," "bestie," "fr") and keep it conversational.
                5. Conciseness: Keep responses short (1-3 sentences when possible) and easy to understand.
                6. Style: Sound like a supportive peer rather than a clinical professional, You should not use emojis at all
                7. Resource Priority: Stevens CAPS first, UWill second, external resources last.

                Remember: You're a friendly guide to Stevens mental health resources, not a therapist. Your goal is to connect students with the right campus support systems while being relatable and approachable.
                """
            
            # Format conversation history for context
            messages = []
            
            # First add the system message
            messages.append({"role": "system", "content": system_message})
            
            # Add up to the last 5 conversation turns for context
            for item in self.conversation_state["history"][-10:]:
                messages.append({"role": item["role"], "content": item["content"]})
            
            # Add the current user query
            messages.append({"role": "user", "content": query})
            
            # Generate the response
            response = client.chat.completions.create(
                model=self.completions_deployment,
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            
            # Extract and return the assistant's response
            assistant_response = response.choices[0].message.content
            return assistant_response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            # Fallback response in case of failure
            return "I'm sorry, I'm having trouble processing your question. Could you please try again? If you're experiencing a mental health emergency, please contact a crisis helpline or emergency services."
        
    def process_user_input(self, user_input: str) -> str:
        """Process user input and generate appropriate response"""
        try:
            # Log user input
            logger.info(f"User input: {user_input}")
            print(f"\n👤 User: {user_input}")
            
            # Check if user is saying goodbye
            if self.is_farewell(user_input):
                farewell_msg = "Thank you for chatting with me. Take care of yourself, and remember that support is always available when you need it. Goodbye!"
                print(f"🧠 Detected farewell message, sending goodbye response")
                
                # Store the farewell exchange in chat history
                store_chat_history(self.session_id, user_input, farewell_msg, self.conversation_state["alert_flag"])
                
                return farewell_msg
            
            # Add user input to conversation history
            self.conversation_state["history"].append({"role": "user", "content": user_input})
            
            # Process user input based on content
            print("🔍 Processing user input...")
            
            # Check if the input needs mental health analysis
            if self.is_expressing_concern(user_input) or len(user_input.split()) > 10:
                if self.is_expressing_concern(user_input):
                    print("🧠 Detected emotional concern in message")
                
                # Analyze for mental health severity
                analysis_result = self.analyze_mental_health_severity(user_input)
                
                # Update alert flag in conversation state
                self.conversation_state["alert_flag"] = analysis_result.get("alert_flag", 0)
                
                # If alert flag is set, trigger alert process
                if self.conversation_state["alert_flag"] == 1:
                    print(f"🚨 Alert flag raised: {analysis_result.get('reason', 'Mental health concern detected')}")
                    self.trigger_alert(analysis_result.get("reason", "Mental health concern detected"))
                
                # Generate response with RAG and mental health consideration
                response = self.generate_response_with_rag(user_input, analysis_result)
            else:
                # For shorter inputs or greetings, use simpler processing
                if self.is_greeting(user_input) and self.conversation_state["is_greeting_phase"]:
                    print("🧠 Detected greeting in initial conversation phase")
                    self.conversation_state["is_greeting_phase"] = False
                    response = "Hey there! I'm Loona, your mental health bestie here at Stevens. What's on your mind today?"
                else:
                    # Generate standard response
                    response = self.generate_response_with_rag(user_input)
            
            # Add response to conversation history
            self.conversation_state["history"].append({"role": "assistant", "content": response})
            
            # Store the exchange in chat history DB
            store_chat_history(self.session_id, user_input, response, self.conversation_state["alert_flag"])
            
            # Maintain conversation history size
            if len(self.conversation_state["history"]) > 20:
                self.conversation_state["history"] = self.conversation_state["history"][-20:]
            
            # Log the response
            logger.info(f"Assistant response: {response}")
            print(f"\n🤖 Assistant: {response}")
            
            return response
            
        except Exception as e:
            error_message = f"Error processing user input: {str(e)}"
            logger.error(error_message)
            print(f"\n❌ Input Processing Error: {str(e)}")
            
            # Even for errors, store the exchange in chat history
            response = "I apologize, but I'm experiencing a technical issue. If you're facing a mental health emergency, please contact a crisis helpline or emergency services immediately."
            store_chat_history(self.session_id, user_input, response, self.conversation_state["alert_flag"])
            
            return response
    
    def run_conversation_loop(self):
        """Run the main conversation loop with VOICE-ONLY interaction"""
        print("\nMental Health Voice Assistant is ready. Please speak.")
        print(f"Session ID: {self.session_id}")
        
        # Initial greeting
        initial_greeting = "Hey there! I'm Loona, your mental health bestie here at Stevens. What's on your mind today?"
        print(f"\n🤖 Assistant: {initial_greeting}")
        self.text_to_speech(initial_greeting)
        
        # Store the initial greeting in chat history
        store_chat_history(self.session_id, "", initial_greeting, 0)
        
        while True:
            try:
                print("\nListening... (Voice only)")
                
                # Use only voice recognition
                user_input = self.recognize_from_microphone()
                
                # Check for no input
                if not user_input.strip():
                    print("No speech detected. Please speak clearly.")
                    continue
                
                # Check for exit command in voice input
                if any(farewell in user_input.lower() for farewell in ["exit", "quit", "bye", "goodbye"]):
                    farewell_msg = "Thank you for using the Mental Health Assistant. Take care!"
                    print(f"\n🤖 Assistant: {farewell_msg}")
                    self.text_to_speech(farewell_msg)
                    
                    # Store the farewell in chat history
                    store_chat_history(self.session_id, user_input, farewell_msg, 0)
                    break
                
                # Process the input and get response
                response = self.process_user_input(user_input)
                
                # Convert response to speech
                speech_success = self.text_to_speech(response)
                if not speech_success:
                    logger.warning("Text-to-speech conversion failed")
                
            except KeyboardInterrupt:
                print("\nExiting the Mental Health Assistant. Take care!")
                break
            except Exception as e:
                logger.error(f"Error in conversation loop: {str(e)}")
                print("I'm sorry, I encountered an error. Let's continue listening.")
    
    def save_conversation_logs(self, filepath="conversation_logs.json"):
        """Save the conversation logs to a file"""
        try:
            with open(filepath, 'w') as f:
                json.dump({
                    "session_id": self.session_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "history": self.conversation_state["history"],
                    "alert_flag": self.conversation_state["alert_flag"]
                }, f, indent=2)
            logger.info(f"Conversation logs saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving conversation logs: {str(e)}")
            return False


# Update the FastAPI endpoint to include chat history
class VoiceRequest(BaseModel):
    text: Optional[str] = None  # in case we want to test text input too

@app.post("/api/voice-chat")
def voice_chat():
    """API endpoint for voice interactions - VOICE ONLY"""
    assistant = MentalHealthVoiceAssistant()
    try:
        # Exclusively use voice recognition
        user_input = assistant.recognize_from_microphone()
        
        # Ensure voice input was captured
        if not user_input:
            return {
                "success": False, 
                "error": "No voice input recognized. Please speak clearly into the microphone."
            }

        # Process the voice input
        response = assistant.process_user_input(user_input)
        
        # Convert response to speech
        speech_success = assistant.text_to_speech(response)
        
        # Return the session info along with the response
        return {
            "success": True, 
            "input": user_input, 
            "response": response,
            "session_id": assistant.session_id,
            "speech_success": speech_success,
            "alert_flag": assistant.conversation_state["alert_flag"]
        }

    except Exception as e:
        logger.error(f"Voice chat API error: {str(e)}")
        return {
            "success": False, 
            "error": f"An error occurred during voice processing: {str(e)}"
        }

@app.get("/api/sessions")
def list_sessions():
    """API endpoint to list all conversation sessions"""
    try:
        cosmos_client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
        database = cosmos_client.get_database_client(HIST_DATABASE_NAME)
        container = database.get_container_client(HIST_CONTAINER_NAME)
        
        # Query for unique session IDs, ordered by most recent first
        query = """
        SELECT DISTINCT c.session_id, MAX(c.timestamp) as last_interaction
        FROM c
        GROUP BY c.session_id
        ORDER BY MAX(c.timestamp) DESC
        """
        
        results = list(container.query_items(
            query=query,
            enable_cross_partition_query=True
        ))
        
        return {"success": True, "sessions": results}
    except Exception as e:
        logger.error(f"Error listing sessions: {str(e)}")
        return {"success": False, "error": str(e)}

@app.get("/api/sessions/{session_id}")
def get_session_history(session_id: str):
    """API endpoint to get conversation history for a specific session"""
    try:
        cosmos_client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
        database = cosmos_client.get_database_client(HIST_DATABASE_NAME)
        container = database.get_container_client(HIST_CONTAINER_NAME)
        
        # Query for messages in this session, ordered by timestamp
        query = f"""
        SELECT c.id, c.timestamp, c.user_input, c.bot_response, c.alert_flag
        FROM c
        WHERE c.session_id = '{session_id}'
        ORDER BY c.timestamp
        """
        
        results = list(container.query_items(
            query=query,
            enable_cross_partition_query=True
        ))
        
        return {"success": True, "session_id": session_id, "messages": results}
    except Exception as e:
        logger.error(f"Error getting session history: {str(e)}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    # Create and run the assistant
    assistant = MentalHealthVoiceAssistant()
    
    try:
        # Run the main conversation loop
        assistant.run_conversation_loop()
    except Exception as e:
        logger.error(f"Critical error in main program: {str(e)}")
    finally:
        # Save logs before exiting
        assistant.save_conversation_logs()
        logger.info("Mental Health Assistant shutting down")
