import json
import re


# ==========================================
# Generate Quiz Questions
# ==========================================

def generate_quiz(client, context, difficulty, num_questions=5):

    prompt = f"""
You are CodeMentor AI, an expert programming tutor.

Generate {num_questions} multiple-choice questions
using ONLY the learning material provided below.

Difficulty level: {difficulty}

Requirements:

1. Each question must have exactly four options.
2. Only one option must be correct.
3. Questions must test understanding, not just memorization.
4. Explanations must explain why the correct answer is right.
5. Do not create questions about information absent from the material.

Return the result as valid JSON only.

Use this exact structure:

{{
    "questions": [
        {{
            "question": "Question text",
            "options": [
                "Option A",
                "Option B",
                "Option C",
                "Option D"
            ],
            "answer": "Correct option",
            "explanation": "Explanation of the correct answer"
        }}
    ]
}}

Learning Material:

{context}
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    if not response.text:
        raise ValueError("Gemini returned an empty quiz response.")

    # Remove Markdown code fences if Gemini includes them
    quiz_text = response.text.strip()

    quiz_text = re.sub(
        r"^```(?:json)?\s*",
        "",
        quiz_text
    )

    quiz_text = re.sub(
        r"\s*```$",
        "",
        quiz_text
    )

    # Convert JSON text into a Python dictionary
    quiz_data = json.loads(quiz_text)

    questions = quiz_data.get("questions", [])

    if len(questions) != num_questions:
        raise ValueError(
            f"Expected {num_questions} questions, "
            f"but received {len(questions)}."
        )

    # Validate question structure
    for question in questions:

        required_fields = [
            "question",
            "options",
            "answer",
            "explanation"
        ]

        for field in required_fields:

            if field not in question:
                raise ValueError(
                    f"Missing field in quiz question: {field}"
                )

        if len(question["options"]) != 4:
            raise ValueError(
                "Every question must contain exactly four options."
            )

        if question["answer"] not in question["options"]:
            raise ValueError(
                "The correct answer must match one of the options."
            )

    return questions