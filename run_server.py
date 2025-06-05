"""
This script starts the Uvicorn server for the Gemini OpenAI Compatible API.

It loads environment variables, configures the server host and port,
and runs the FastAPI application defined in `api_server.py`.
Auto-reload is enabled for development purposes.
"""

import os
import uvicorn
from dotenv import load_dotenv

def main():
    """
    Loads configuration and starts the Uvicorn server.
    
    Server host, port, and other settings can be configured via environment
    variables (see .env.example or README.md).
    """
    # Load environment variables
    load_dotenv()
    
    # Get server configuration from environment or use defaults
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    # Configure and start the server
    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        reload=True,  # Set to False in production or use an env var
        log_level="info" # Consider making this configurable via env var
    )

if __name__ == "__main__":
    main() 