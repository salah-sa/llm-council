"""OpenRouter API client for making LLM requests."""

import httpx
from typing import List, Dict, Any, Optional
from .config import OPENROUTER_API_KEY, OPENROUTER_API_URL


async def query_model(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = 120.0,
    max_retries: int = 3
) -> Optional[Dict[str, Any]]:
    """
    Query a single model via OpenRouter API with retry-on-429.

    Args:
        model: OpenRouter model identifier (e.g., "openai/gpt-4o")
        messages: List of message dicts with 'role' and 'content'
        timeout: Request timeout in seconds
        max_retries: Number of retries on 429 rate-limit errors

    Returns:
        Response dict with 'content' and optional 'reasoning_details', or None if failed
    """
    import asyncio

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
    }

    for attempt in range(max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    OPENROUTER_API_URL,
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()

                data = response.json()

                # Guard: some models return 200 with error body (no 'choices')
                if 'choices' not in data or not data['choices']:
                    error_msg = data.get('error', {}).get('message', str(data)[:200])
                    print(f"[OpenRouter] No choices in response for {model}: {error_msg}")
                    return None

                message = data['choices'][0]['message']

                return {
                    'content': message.get('content'),
                    'reasoning_details': message.get('reasoning_details')
                }

        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            body = e.response.text[:500] if e.response else "no body"

            if status == 429 and attempt < max_retries:
                wait = 5 * (2 ** attempt)  # 5s, 10s, 20s
                print(f"[OpenRouter] 429 rate-limited for {model}, retrying in {wait}s (attempt {attempt+1}/{max_retries})...")
                await asyncio.sleep(wait)
                continue

            print(f"[OpenRouter] HTTP {status} for {model}: {body}")
            return None
        except Exception as e:
            print(f"[OpenRouter] Error querying model {model}: {type(e).__name__}: {e}")
            return None

    return None


async def query_models_parallel(
    models: List[str],
    messages: List[Dict[str, str]]
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple models in parallel.

    Args:
        models: List of OpenRouter model identifiers
        messages: List of message dicts to send to each model

    Returns:
        Dict mapping model identifier to response dict (or None if failed)
    """
    import asyncio

    # Create tasks for all models
    tasks = [query_model(model, messages) for model in models]

    # Wait for all to complete
    responses = await asyncio.gather(*tasks)

    # Map models to their responses
    return {model: response for model, response in zip(models, responses)}
