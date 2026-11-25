import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("API key not found!")
    exit()

print("API key loaded.")

client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(timeout=20000)
)

print("Sending request to Gemini...")

try:
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents="Explain Python variables in two sentences."
    )

    print("\nGemini Response:")
    print(response.text)

except Exception as e:
    print("\nError occurred:")
    print(type(e).__name__)
    print(e)