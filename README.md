# Gemini Web OpenAI Proxy

An unofficial OpenAI-compatible REST API interface for Google's Gemini, powered by gemini-webapi (browser-based). Enables using Gemini through OpenAI-compatible tools and SDKs without requiring API keys.

## Features

- OpenAI-compatible endpoints:
    - `/v1/chat/completions` (supports streaming)
    - `/v1/models`
- Simulated streaming for chat completions to mimic OpenAI's SSE behavior.
- Asynchronous implementation using FastAPI for good performance.
- CORS support for web applications.
- Configuration via environment variables.
- Supports multiple Gemini models (see "Available Models" below).

## Setup

### Option 1: Local Setup

1.  **Prerequisites**:
    * Python 3.8+
    * Pip

2.  **Clone the repository (if you haven't already):**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Create a `.env` file:**
    Copy the `.env.example` file (if provided) to `.env`, or create a new `.env` file in the project root with your Gemini credentials:
    ```env
    # Required: Gemini credentials from your browser session
    # Ensure these are kept secure and are refreshed as needed.
    SECURE_1PSID=your_secure_1psid_here
    SECURE_1PSIDTS=your_secure_1psidts_here

    # Optional: Server configuration
    HOST=0.0.0.0  # Default host to bind the server
    PORT=8000     # Default port for the server
    ```
    *Obtaining `SECURE_1PSID` and `SECURE_1PSIDTS`*: These cookies are typically obtained by logging into your Google account in a web browser.
    - Go to https://gemini.google.com and login with your Google account
    - Press F12 for web inspector, go to Network tab and refresh the page
    - Click any request and copy cookie values of `__Secure-1PSID` and `__Secure-1PSIDTS`

5.  **Start the server:**
    ```bash
    # Method 1 (Recommended): Using the run script
    python run_server.py

    # Method 2: Using uvicorn directly (e.g., for more Uvicorn options)
    # uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
    ```

### Option 2: Docker Setup

1. **Prerequisites**:
   * Docker installed on your system

2. **Build the Docker image:**
   ```bash
   docker build -t gemini-openai-api .
   ```

3. **Create a `.env` file** as described in Option 1, Step 4.

4. **Run the container:**
   ```bash
   docker run -d \
     --name gemini-api \
     -p 8000:8000 \
     --env-file .env \
     gemini-openai-api
   ```

   This will:
   - Run the container in detached mode (-d)
   - Name it "gemini-api"
   - Map port 8000 on your host to port 8000 in the container
   - Load environment variables from your .env file
   - Run using uvicorn server directly for better production performance

   You can customize the uvicorn settings by passing environment variables:
   ```bash
   docker run -d \
     --name gemini-api \
     -p 8000:8000 \
     -e HOST=0.0.0.0 \
     -e PORT=8000 \
     --env-file .env \
     gemini-openai-api
   ```

5. **View logs:**
   ```bash
   docker logs -f gemini-api
   ```

6. **Stop the container:**
   ```bash
   docker stop gemini-api
   ```

The server will start and listen on `http://localhost:8000` (or your configured host/port). The `--reload` flag is useful for development as it automatically restarts the server on code changes.

## API Usage

### List Models

Retrieve a list of available models.

```bash
curl http://localhost:8000/v1/models
```

### Chat Completion

Generate a response from a model.

**Non-Streaming:**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-2.5-flash",
    "messages": [
      {"role": "user", "content": "Hello! How are you today?"}
    ]
  }'
```

**Streaming:**
To receive a streamed response, add `"stream": true` to your request.
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-2.5-pro",
    "messages": [
      {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    "stream": true
  }'
```
The response will be a Server-Sent Events (SSE) stream.

## OpenAI SDK Compatibility

You can use this API with the OpenAI Python SDK (and likely other OpenAI-compatible libraries) by setting the `base_url` and providing a placeholder API key.

```python
from openai import AsyncOpenAI # Or `OpenAI` for synchronous usage with GeminiOpenAISync (not directly exposed via server)

# For asynchronous client (recommended with this async server)
client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # API key is not used by this server but often required by the SDK
)

async def get_completion():
    response = await client.chat.completions.create(
        model="gemini-2.5-pro", # Or any other available model
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ]
    )
    print(response.choices[0].message.content)

async def get_streaming_completion():
    stream = await client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {"role": "user", "content": "Write a short poem about stars."}
        ],
        stream=True
    )
    async for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")
    print() # For a newline after the stream

# Example usage with asyncio
# import asyncio
# asyncio.run(get_completion())
# asyncio.run(get_streaming_completion())
```

## Available Models

The API currently exposes the following models (identifiers to be used in the `model` field of requests):

-   `gemini-2.5-flash`: A fast and efficient model for general tasks.
-   `gemini-2.0-flash`: An earlier version of the flash model.
-   `gemini-2.0-flash-thinking`: (Purpose may vary, potentially for reasoning tasks)
-   `gemini-2.5-pro`: A more capable model for complex tasks.

The specific capabilities (e.g., text, vision) depend on the underlying Gemini model version accessed by `gemini-webapi`. This API primarily focuses on text-based chat completions.

## Limitations

-   **Token Counting**: Token count information (`usage` field in the response) is returned as `0` because the underlying `gemini-webapi` does not currently provide this data.
-   **Streaming**: Streaming is *simulated*. The server first gets the full response from Gemini and then sends it in chunks. This provides compatibility but isn't "true" streaming from the core model.
-   **Parameter Mapping**: Parameters like `temperature` and `max_tokens` are accepted by the API but may not be fully honored or passed to the underlying Gemini model via `gemini-webapi`, as noted in the code. Their effect might be limited.
-   **Error Handling**: Error messages from the underlying `gemini-webapi` are passed through but might not always map perfectly to OpenAI error codes or formats.
-   **Authentication**: Relies on `SECURE_1PSID` and `SECURE_1PSIDTS` cookies, which can expire and require manual refresh. This method is suitable for personal use or development, not robust production deployments.
-   **Feature Parity**: Not all features of the official OpenAI API (e.g., function calling, embeddings, fine-tuning) are implemented. This API focuses on core chat completion and model listing.

## API Documentation

Once the server is running, interactive API documentation is available at:

-   **Swagger UI**: `http://localhost:8000/docs`
-   **ReDoc**: `http://localhost:8000/redoc`

These interfaces allow you to explore and test the API endpoints directly from your browser. 