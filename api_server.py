import asyncio
import json
import os
import time
from typing import List, Optional, Dict, Any, Union
import uuid
from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from gemini_openai import GeminiOpenAI
# Note: Model from gemini_webapi.constants is used by GeminiOpenAI, not directly here for model listing
# from gemini_webapi.constants import Model

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Gemini OpenAI Compatible API",
    description="A REST API that provides OpenAI-compatible endpoints using Google's Gemini model",
    version="1.0.0",
    docs_url="/docs", # Serve docs at /docs
    redoc_url="/redoc"  # Serve redoc at /redoc
)

# API router for versioning
router_v1 = APIRouter(prefix="/v1")


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response validation
class Message(BaseModel):
    role: str = Field(..., description="The role of the message sender (system, user, or assistant)")
    content: str = Field(..., description="The content of the message")

class ChatCompletionRequest(BaseModel):
    model: str = Field(default="gemini-2.5-flash", description="The model to use for completion")
    messages: List[Message] = Field(..., description="The messages to generate a response for")
    temperature: Optional[float] = Field(default=0.7, description="The sampling temperature (Note: May not be fully supported by the underlying Gemini model via this API)")
    max_tokens: Optional[int] = Field(default=None, description="The maximum number of tokens to generate (Note: May not be fully supported by the underlying Gemini model via this API)")
    stream: Optional[bool] = Field(default=False, description="Whether to stream the response")

# Global client instance
client: Optional[GeminiOpenAI] = None
_client_lock = asyncio.Lock()


async def get_client() -> GeminiOpenAI:
    """
    Retrieves an initialized GeminiOpenAI client, creating it if necessary.

    This function uses an asyncio.Lock to prevent race conditions during
    concurrent initialization attempts of the global client.
    """
    global client
    if client is None:
        async with _client_lock:
            if client is None: # Double check after acquiring lock
                secure_1psid = os.getenv("SECURE_1PSID")
                secure_1psidts = os.getenv("SECURE_1PSIDTS")
                print(f"SECURE_1PSID: {'Set' if secure_1psid else 'Not Set'}", 
                      f"SECURE_1PSIDTS: {'Set' if secure_1psidts else 'Not Set'}")
                if not secure_1psid or not secure_1psidts:
                    raise HTTPException(
                        status_code=500,
                        detail="Missing Gemini credentials. Please set SECURE_1PSID and SECURE_1PSIDTS environment variables."
                    )
                
                temp_client = GeminiOpenAI(secure_1psid, secure_1psidts)
                await temp_client._ensure_initialized() # Initialize before assigning to global
                client = temp_client
    return client

async def simulate_streaming_response_async(full_response: dict, delay=0.005):
    """
    Simulates an OpenAI-like streaming response by breaking down a complete
    chat completion response into chunks.

    This function takes a fully generated response (as if `stream=False` was used)
    and yields it piece by piece to emulate the behavior of a streaming API,
    making it compatible with clients expecting Server-Sent Events (SSE).

    Args:
        full_response: The complete response dictionary from `GeminiOpenAI.chat_completion`.
                       This dictionary is expected to be in OpenAI's chat completion format.
        delay: A small delay (in seconds) between yielding chunks. This is primarily
               for simulation purposes and to avoid overwhelming the client.
    """
    # Extract metadata from the full_response once, with robust defaults
    chat_id = full_response.get('id', f'chatcmpl-simulated-{uuid.uuid4()}')
    created_timestamp = full_response.get('created', int(time.time()))
    # Ensure model_name is always a string, even if 'model' is missing in full_response
    model_name = str(full_response.get('model', 'gpt-4o-simulated')) # Use a distinct name for simulated model

    # Extract role from the full response message
    role = "assistant" # Default role
    if full_response.get("choices") and \
       isinstance(full_response["choices"], list) and \
       len(full_response["choices"]) > 0 and \
       isinstance(full_response["choices"][0], dict) and \
       full_response["choices"][0].get("message") and \
       isinstance(full_response["choices"][0]["message"], dict):
        role = str(full_response["choices"][0]["message"].get("role", "assistant"))

    # 1. Send the initial chunk with role
    initial_chunk_data = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created_timestamp,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": {"role": role},
                # "finish_reason" is omitted as it's not determined yet
            }
        ]
        # "system_fingerprint": "fp_simulated" # Optional: sometimes clients look for this
    }
    yield f"data: {json.dumps(initial_chunk_data)}\n\n"
    if delay > 0: await asyncio.sleep(delay)


    # 2. Stream content character by character
    content = ""
    if full_response.get("choices") and \
       isinstance(full_response["choices"], list) and \
       len(full_response["choices"]) > 0 and \
       isinstance(full_response["choices"][0], dict) and \
       full_response["choices"][0].get("message") and \
       isinstance(full_response["choices"][0]["message"], dict):
        content = str(full_response["choices"][0]["message"].get("content", ""))

    if not content: # If content is empty, send an empty content chunk before finish
        empty_content_chunk_data = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created_timestamp,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": ""}, # Explicit empty content
                }
            ]
        }
        yield f"data: {json.dumps(empty_content_chunk_data)}\n\n"
        if delay > 0: await asyncio.sleep(delay)
    else:
        for char_index, char in enumerate(content):
            char_chunk_data = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created_timestamp,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": str(char)}, # Ensure char is a string
                        # "finish_reason" is omitted
                    }
                ]
            }
            yield f"data: {json.dumps(char_chunk_data)}\n\n"
            if delay > 0: await asyncio.sleep(delay)

    # 3. Send the final chunk with finish_reason
    finish_reason_val = "stop" # Default finish reason
    if full_response.get("choices") and \
       isinstance(full_response["choices"], list) and \
       len(full_response["choices"]) > 0 and \
       isinstance(full_response["choices"][0], dict) and \
       full_response["choices"][0].get("finish_reason"): # Check if key exists
        finish_reason_val = str(full_response["choices"][0].get("finish_reason", "stop"))


    final_chunk_data = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created_timestamp,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": {}, # Empty delta for the final chunk
                "finish_reason": finish_reason_val
            }
        ]
    }
    yield f"data: {json.dumps(final_chunk_data)}\n\n"
    
    # 4. Send [DONE] signal
    yield "data: [DONE]\n\n"
    
@router_v1.post("/chat/completions", response_model_exclude_none=True)
async def create_chat_completion(request: ChatCompletionRequest): # Removed Dict[str, Any] for StreamingResponse
    """
    Creates a completion for the chat message. This endpoint is compatible
    with OpenAI's chat completion endpoint (`/v1/chat/completions`).

    It supports both regular (full response) and streaming responses.
    If `request.stream` is true, it will simulate a stream using Server-Sent Events.
    """
    try:
        gemini_client = await get_client() # Renamed variable for clarity
        
        messages = [msg.dict() for msg in request.messages]
        response = await gemini_client.chat_completion(
            messages=messages,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=request.stream # Pass stream to client, though client doesn't stream internally from gemini
        )

        if request.stream:
            return StreamingResponse(simulate_streaming_response_async(response), media_type="text/event-stream")
        else:
            return response
    except HTTPException: # Re-raise HTTPExceptions directly
        raise
    except Exception as e:
        print(f"Error in create_chat_completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@router_v1.get("/models")
async def list_models() -> Dict[str, Any]:
    """
    Lists the available models compatible with this API.
    This endpoint is compatible with OpenAI's models endpoint (`/v1/models`).

    The model list is currently hardcoded but should reflect the models
    supported by the `GeminiOpenAI` client's `_model_mapping`.
    """
    # Note: This list should be kept in sync with _model_mapping in gemini_openai.py
    # For a more dynamic approach, this could fetch keys from an initialized client instance,
    # but that adds complexity to this simple endpoint.
    supported_models = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-2.0-flash-thinking",
        "gemini-2.5-pro"
    ]
    
    models_data = []
    for model_id in supported_models:
        models_data.append({
            "id": model_id,
            "object": "model",
            "created": int(time.time()), # Using current time as placeholder
            "owned_by": "google",
            "permission": [], # Placeholder
            "root": model_id,
            "parent": None # Placeholder
        })
        
    return {
        "data": models_data,
        "object": "list"
    }

app.include_router(router_v1) # Include the v1 router

@app.on_event("shutdown")
async def shutdown_event():
    """
    Performs cleanup actions when the FastAPI application is shutting down.
    Specifically, it closes the `GeminiOpenAI` client connection if it was initialized.
    """
    global client
    if client:
        await client.close()
        client = None 