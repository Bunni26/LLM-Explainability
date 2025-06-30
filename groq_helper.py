import os
import requests
from dotenv import load_dotenv
from pathlib import Path

# ✅ Explicitly load from root .env file
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables!")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

VALID_MODELS = {
    "llama3-8b-8192",
    "llama3-70b-8192",
    "mixtral-8x7b-32768",
    "gemma-7b-it"
}

def llama_infer(prompt, model="llama3-8b-8192", temperature=0.7, max_tokens=512):
    if model not in VALID_MODELS:
        raise ValueError(f"Model '{model}' is not supported by Groq. Choose from: {', '.join(VALID_MODELS)}")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful, explainable LLM."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=body)
        response.raise_for_status()
        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"].strip()
        else:
            raise ValueError("No valid response content found in API response")
    except requests.exceptions.RequestException as e:
        raise Exception(f"API request failed: {str(e)}")
    except (KeyError, IndexError) as e:
        raise Exception(f"Error processing API response: {str(e)}")
    except Exception as e:
        raise Exception(f"Unexpected error: {str(e)}")