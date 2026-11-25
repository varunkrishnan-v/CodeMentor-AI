import os

from dotenv import load_dotenv
from google import genai

from debugger import debug_python_code


# Load API key from .env
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)


# Python code containing an error
sample_code = """
numbers = [10, 20, 30, 40]

total = 0

for i in range(len(numbers)):
    total = total + numbers[i]

print(totl)
"""


# Select explanation difficulty
difficulty = "Beginner"


# Call debugging assistant
result = debug_python_code(
    client=client,
    code=sample_code,
    difficulty=difficulty
)


# Display result
print("\n========== DEBUGGING REPORT ==========\n")

print(result)