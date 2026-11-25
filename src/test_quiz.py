import os

from dotenv import load_dotenv
from google import genai

from quiz_generator import generate_quiz


# ==========================================
# Load Gemini API
# ==========================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(
    api_key=api_key
)


# ==========================================
# Sample Learning Material
# ==========================================

context = """
Python is a programming language.

A variable is used to store data.

The print() function displays output on the screen.

Python supports different data types such as
integers, strings, floats and booleans.

A list is used to store multiple items in a
single variable. Lists are mutable, meaning
their elements can be changed.
"""


# ==========================================
# Generate Quiz
# ==========================================

questions = generate_quiz(
    client=client,
    context=context,
    difficulty="Beginner",
    num_questions=5
)


# ==========================================
# Display Quiz
# ==========================================

for i, question in enumerate(questions, start=1):

    print(f"\nQuestion {i}: {question['question']}")

    for option in question["options"]:

        print("-", option)

    print("Correct Answer:", question["answer"])

    print("Explanation:", question["explanation"])