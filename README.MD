
# Campus Calm - AI Mental Health Assistant for Stevens Institute of Technology

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <!-- Add other relevant badges -->

Campus Calm is an AI-powered mental health assistant specifically designed for students at Stevens Institute of Technology. It provides a supportive and confidential space for students to discuss their concerns, access relevant campus resources, and receive immediate crisis information when needed. The assistant leverages Azure AI services, including OpenAI for language understanding and generation, Cosmos DB for knowledge retrieval (RAG) and conversation history, and Azure Speech Services for voice interaction.

![Alt text](/tThumbnail.png?raw=true "Title")


## Overview

This application serves as a dual-interface (text and voice) chatbot. It uses a Retrieval-Augmented Generation (RAG) approach, primarily drawing information from the Stevens Institute of Technology "Red Folder" mental health protocol to provide contextually relevant and accurate information about campus resources. It also includes a severity analysis feature to detect potential crises and trigger email alerts to administrators for timely intervention.

## Key Features

*   **Dual Interface:** Supports both text-based chat (`/api/text-chat`) and voice-based interaction (`/api/voice-chat`).
*   **Stevens-Specific Knowledge:** Utilizes RAG with information sourced from the Stevens Red Folder and other relevant documents stored in Azure Cosmos DB.
*   **Voice Interaction:** Leverages Azure Cognitive Speech Services for accurate speech-to-text and natural-sounding text-to-speech (using `en-US-JennyNeural`).
*   **Mental Health Severity Analysis:** Analyzes user input to detect signs of distress, potential self-harm, or crisis situations using predefined keywords and AI models (potentially Azure AI Studio Agents).
*   **Automated Alerts:** Triggers email notifications to designated administrators via SMTP when severe mental health concerns are detected.
*   **Conversation History:** Stores and retrieves chat history per session using Azure Cosmos DB.
*   **Scalable Backend:** Built with FastAPI and Uvicorn for efficient asynchronous request handling.
*   **Configurable:** Uses environment variables for easy configuration of Azure services and alert settings.
*   **Gen Z Persona:** Aims for a supportive, casual, and relatable tone ("Loona") suitable for students.

## Architecture

The application follows a modular design:

1.  **Frontend (Not Included):** A separate web application (likely React, based on `index.css`/`App.css`) interacts with the backend API endpoints.
2.  **FastAPI Backend (`main.py`):**
    *   Acts as the main entry point and API router.
    *   Handles incoming HTTP requests for text and voice chat.
    *   Manages CORS (Cross-Origin Resource Sharing).
    *   Provides endpoints for retrieving session history.
    *   Initializes and delegates tasks to the appropriate bot module.
3.  **Text Bot Module (`updated_trag_ch.py` - `IntegratedMentalHealthBot`):**
    *   Handles text-based conversation logic.
    *   Performs RAG using vector search (or keyword fallback) against the Cosmos DB knowledge base.
    *   Integrates the hardcoded `red_folder` content into prompts.
    *   Interacts with Azure OpenAI (`gpt-4o`, `text-embedding-3-small`) for response generation and embeddings.
    *   Includes logic for mental health severity analysis and triggering alerts via the `trigger_alert` function.
    *   Manages its own conversation state (history, alert flag).
4.  **Voice Bot Module (`updated_vrag.py` - `MentalHealthVoiceAssistant`):**
    *   Handles voice-based conversation logic.
    *   Uses Azure Speech SDK for speech recognition and synthesis.
    *   Also performs mental health severity analysis and triggers alerts (similar logic to the text bot).
    *   Integrates the hardcoded `red_folder` content.
    *   Manages its own conversation state and a unique `session_id`.
    *   Includes the primary logic for storing chat history (`store_chat_history`) in a separate Cosmos DB container.
5.  **Azure Services:**
    *   **Azure OpenAI:** Provides embedding models (e.g., `text-embedding-3-small`) and large language models (e.g., `gpt-4o`) for RAG and chat completion.
    *   **Azure Cosmos DB (SQL API):**
        *   Used as a vector database for the RAG knowledge base (documents, metadata, embeddings).
        *   Used as a NoSQL database to store conversation history (session ID, timestamp, user/bot messages, alert flag).
    *   **Azure Cognitive Speech Services:** Enables voice input (speech-to-text) and output (text-to-speech).
    *   **Azure AI Studio Agents (Implied):** Potentially used via `AGENT_ID` and `ALERT_AGENT_ID` for more sophisticated severity analysis or alert generation, though direct interaction logic might be abstracted within Azure services.
6.  **Email Service (SMTP):** Used to send alert emails (e.g., Gmail SMTP).

## Technology Stack

*   **Backend:** Python 3.x, FastAPI, Uvicorn
*   **AI/ML:** Azure OpenAI (GPT-4o, Embeddings), Azure Cognitive Speech Services
*   **Database:** Azure Cosmos DB (SQL API with Vector Search)
*   **Libraries:**
    *   `openai` (Azure SDK)
    *   `azure-cosmos`
    *   `azure-cognitiveservices-speech`
    *   `azure-identity` (likely for AI Studio Agent auth)
    *   `azure-ai-projects` (likely for AI Studio Agent interaction)
    *   `pydantic` (Data validation)
    *   `python-dotenv` (Environment variables)
    *   `smtplib`, `email` (Email alerts)
    *   `requests`, `httpx` (HTTP clients)
    *   `langchain-*` (Dependencies present, direct usage in provided code seems minimal - potentially used by underlying agents or for specific splitting/processing not shown)
*   **Deployment:** (Assumed) Azure App Service, Azure Kubernetes Service, or Virtual Machines.

## Prerequisites

*   Python 3.8+
*   An Azure Subscription with access to:
    *   Azure OpenAI Service (with deployments for GPT-4o and text-embedding-3-small)
    *   Azure Cosmos DB (SQL API account, with vector search enabled if using)
    *   Azure Cognitive Speech Services
    *   (Potentially) Azure AI Studio project with configured Agents (for `AGENT_ID`, `ALERT_AGENT_ID`)
*   An email account configured for SMTP sending (e.g., Gmail with an App Password).
*   `git` (for cloning the repository).

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

1.  **Create a `.env` file:** Copy the contents of `env(sample).txt` into a new file named `.env` in the project's root directory.
    ```bash
    cp env\(sample\).txt .env
    ```
    *(Note: Adjust the copy command based on your shell if the above fails due to parentheses).*

2.  **Fill in the `.env` file:** Populate the `.env` file with your actual credentials and endpoints for all the Azure services and email configuration.

    **Important Notes on Environment Variables:**
    *   There are duplicate variable names (e.g., `AZURE_OPENAI_ENDPOINT` vs `OPENAI_API_ENDPOINT`). Ensure you fill the ones actively used by the `updated_trag_ch.py` and `updated_vrag.py` scripts or fill both sets consistently. Check the `os.getenv()` calls in the Python files to confirm which ones are needed.
    *   **Cosmos DB:** You need connection details for *two* potential databases/containers: one for the RAG knowledge base (`COSMOS_DATABASE_NAME`, `COSMOS_CONTAINER_NAME`) and one for chat history (`HIST_DATABASE_NAME`, `HIST_CONTAINER_NAME`). Ensure these are created in your Azure Cosmos DB account.
    *   **Azure AI Agents:** If using `AGENT_ID` and `ALERT_AGENT_ID`, ensure the `AZURE_CONN_STRING` is correctly formatted for the `AIProjectClient`. Authentication might rely on `DefaultAzureCredential`, so ensure your environment (e.g., Azure CLI login, Managed Identity) is set up for authentication.
    *   **Email:** For Gmail, you'll likely need to enable 2FA and generate an "App Password" to use in `SENDER_PASSWORD`.

## Running the Application

Once dependencies are installed and the `.env` file is configured, start the FastAPI server using Uvicorn:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

*   `--reload`: Enables auto-reloading when code changes (useful for development).
*   `--host 0.0.0.0`: Makes the server accessible on your local network.
*   `--port 8000`: Specifies the port to run on.

The API will be available at `http://localhost:8000`. You can access the interactive documentation (Swagger UI) at `http://localhost:8000/docs`.

## API Endpoints

The following endpoints are defined in `main.py`:

*   **`GET /`**
    *   Description: Root endpoint, redirects to documentation.
    *   Response: Basic welcome message.
*   **`GET /health`**
    *   Description: Simple health check.
    *   Response: `{"status": "ok", "version": "1.0.0"}`
*   **`POST /api/text-chat`**
    *   Description: Handles text-based chat messages.
    *   Request Body: `{"message": "user's message", "session_id": "optional_session_id"}`
    *   Response: `{"success": true/false, "response": "bot's reply", "alert_flag": 0/1, "error": "error message if success=false"}`
*   **`POST /api/voice-chat`**
    *   Description: Handles voice-based interactions. Primarily uses microphone input but can accept text as fallback.
    *   Request Body (Optional): `{"text": "user's message if not using mic", "session_id": "optional_session_id"}`
    *   Response: `{"success": true/false, "input": "recognized text", "response": "bot's reply", "session_id": "current_session_id", "speech_success": true/false, "alert_flag": 0/1, "error": "error message if success=false"}`
*   **`GET /api/sessions`**
    *   Description: Lists all unique conversation session IDs stored in the history database, ordered by most recent activity.
    *   Response: `{"success": true/false, "sessions": [{"session_id": "...", "last_activity": "timestamp"}, ...], "error": "..."}`
*   **`GET /api/sessions/{session_id}`**
    *   Description: Retrieves the full message history for a specific session ID.
    *   Response: `{"success": true/false, "session_id": "...", "messages": [{"id": "...", "timestamp": "...", "user_input": "...", "bot_response": "...", "alert_flag": 0/1}, ...], "error": "..."}`

## Key Components Explained

*   **`main.py`:** The orchestrator. Initializes FastAPI, sets up middleware, defines API routes, and delegates requests to the appropriate bot instance.
*   **`updated_trag_ch.py` (`IntegratedMentalHealthBot`):** Focuses on the text-based RAG chat logic. Handles embedding generation, vector search in Cosmos DB, constructing prompts with context (including the hardcoded `red_folder`), interacting with Azure OpenAI for chat completions, and performing severity analysis/alerts.
*   **`updated_vrag.py` (`MentalHealthVoiceAssistant`):** Focuses on voice interaction using the Azure Speech SDK. Contains similar logic for severity analysis and alerts but is tailored for voice input/output. Crucially, it also contains the `store_chat_history` function used by the API to save conversation logs to a dedicated Cosmos DB container.
*   **`red_folder` (Variable):** This large multi-line string, hardcoded in both bot files, contains the core text from the Stevens Institute of Technology Red Folder. It's injected directly into the LLM prompts to provide highly relevant, Stevens-specific context for answering questions about resources and protocols. *Note: Hardcoding this is not ideal for maintainability; consider externalizing it to a configuration file or the RAG database.*
*   **Alert System:** Both bots implement logic (`is_expressing_concern`, `analyze_mental_health_severity`, `trigger_alert`, `send_alert_email`, `send_fallback_alert_email`) to assess user input for crisis indicators. If the `alert_flag` is set to 1, an email notification is sent via SMTP to the configured `ADMIN_EMAIL`. The system attempts to use a dedicated Azure AI Agent (`ALERT_AGENT_ID`) to format the email but includes a fallback mechanism.
*   **RAG (Retrieval-Augmented Generation):** The text bot uses vector search on embeddings stored in Cosmos DB to find relevant document chunks based on the user's query. This context, along with the `red_folder` text and conversation history, is fed to the LLM to generate an informed response. Keyword search (`basic_search`) acts as a fallback.
*   **Conversation History:** Managed in two ways:
    *   In-memory within each bot's `conversation_state["history"]`.
    *   Persistently stored in Cosmos DB via the `store_chat_history` function (called by the API endpoints using logic from `updated_vrag.py`).

## Frontend Integration

This repository provides the backend API. A separate frontend application (e.g., built with React, Vue, Angular) would be required to interact with these API endpoints:

1.  Send user messages (text or trigger voice input) to `/api/text-chat` or `/api/voice-chat`.
2.  Display the bot's `response`.
3.  Handle voice input/output if using the voice endpoint.
4.  Potentially use the `alert_flag` in the response to modify the UI (e.g., display a warning or specific resources).
5.  (Optional) Implement an admin dashboard using `/api/sessions` and `/api/sessions/{session_id}` to review conversations, especially those flagged with alerts.

## Contributing

Contributions are welcome! Please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some feature'`).
5.  Push to the branch (`git push origin feature/your-feature-name`).
6.  Open a Pull Request.

Please ensure your code adheres to standard Python best practices and includes relevant comments.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (assuming you add an MIT license file).

## Acknowledgements

*   Stevens Institute of Technology for the "Red Folder" content.
*   Microsoft Azure for providing the AI and cloud services.
*   OpenAI for the underlying language models.
*   The FastAPI and Pydantic communities.
```

**Next Steps:**

1.  **Save:** Save this content as `README.md` in the root of your GitHub repository.
2.  **License File:** If you choose the MIT license (or another), create a file named `LICENSE` in the root directory and paste the corresponding license text into it.
3.  **Customize:**
    *   Replace `<your-repository-url>` and `<your-repository-directory>` with your actual repository details.
    *   Add any other relevant badges (e.g., build status, code coverage) if applicable.
    *   Refine the architecture diagram description if needed.
    *   Clarify the usage of the duplicate environment variables based on your final code.
    *   Add specific details about the Azure AI Studio Agents if you have them.
4.  **Push:** Commit the `README.md` (and `LICENSE` file) and push it to your GitHub repository.
