import time
from typing import List, Optional, Dict, Any, Union
import asyncio
import uuid
from gemini_webapi import GeminiClient
from gemini_webapi.constants import Model

class GeminiOpenAI:
    """
    An asynchronous client for interacting with the Gemini WebAPI using an
    OpenAI-compatible interface.
    """
    def __init__(self, secure_1psid: str, secure_1psidts: str, proxy: Optional[str] = None):
        """
        Initializes the GeminiOpenAI client.

        Args:
            secure_1psid: The SECURE_1PSID cookie value.
            secure_1psidts: The SECURE_1PSIDTS cookie value.
            proxy: Optional proxy server URL.
        """
        self.client = GeminiClient(secure_1psid, secure_1psidts, proxy=proxy)
        self._initialized = False
        self._model_mapping = {
            "gemini-2.5-flash": Model.G_2_5_FLASH,
            "gemini-2.0-flash": Model.G_2_0_FLASH,
            "gemini-2.0-flash-thinking": Model.G_2_0_FLASH_THINKING,
            "gemini-2.5-pro": Model.G_2_5_PRO
        }
        
    async def _ensure_initialized(self):
        """Ensures the GeminiClient is initialized before making requests."""
        if not self._initialized:
            await self.client.init(timeout=300, auto_close=False, close_delay=300, auto_refresh=True)
            self._initialized = True

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gemini-2.5-flash",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        Generates a chat completion response.

        Args:
            messages: A list of messages in OpenAI format.
            model: The model name to use (e.g., "gemini-2.5-flash").
            temperature: Sampling temperature. (Note: Currently not passed to gemini-webapi)
            max_tokens: Maximum tokens to generate. (Note: Currently not passed to gemini-webapi)
            stream: Whether to stream the response. (Note: This implementation returns a full response;
                    streaming is handled at the API server level by simulating chunks from this full response).

        Returns:
            A dictionary containing the chat completion response in OpenAI format.
        """
        await self._ensure_initialized()
        
        prompt = ""
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "system":
                prompt += f"System: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
            elif role == "user":
                prompt += f"User: {content}\n"
        
        gemini_model = self._model_mapping.get(model, Model.G_2_5_FLASH)
        
        response = await self.client.generate_content(
            prompt,
            model=gemini_model,
        )
        current_timestamp = int(time.time())

        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": current_timestamp,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response.text,
                    },
                    "finish_reason": "stop"
                }
            ],  
            "usage": {
                "prompt_tokens": 0,  # Gemini API does not provide token counts
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }

    async def close(self):
        """Close the Gemini client connection."""
        if self._initialized:
            await self.client.close()
            self._initialized = False

# Synchronous wrapper for easier usage
class GeminiOpenAISync:
    """
    A synchronous wrapper for the GeminiOpenAI client, providing a blocking interface.
    """
    def __init__(self, secure_1psid: str, secure_1psidts: str, proxy: Optional[str] = None):
        """
        Initializes the synchronous GeminiOpenAI client.

        Args:
            secure_1psid: The SECURE_1PSID cookie value.
            secure_1psidts: The SECURE_1PSIDTS cookie value.
            proxy: Optional proxy server URL.
        """
        self.async_client = GeminiOpenAI(secure_1psid, secure_1psidts, proxy)
        
    def chat_completion(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Generates a chat completion response synchronously.

        Refer to `GeminiOpenAI.chat_completion` for argument details.

        Returns:
            A dictionary containing the chat completion response in OpenAI format.
        """
        return asyncio.run(self.async_client.chat_completion(*args, **kwargs))
        
    def close(self):
        """Closes the underlying asynchronous client connection."""
        asyncio.run(self.async_client.close()) 