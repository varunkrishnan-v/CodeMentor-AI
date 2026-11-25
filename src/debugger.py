# ==========================================
# Debugging Assistant
# ==========================================

def debug_python_code(client, code, difficulty):

    prompt = f"""
You are CodeMentor AI, an expert Python debugging assistant.

The user has selected this explanation level: {difficulty}

Analyze the following Python code carefully.

Your responsibilities:

1. Identify syntax errors, if any.
2. Identify logical errors, if any.
3. Identify potential runtime errors.
4. Explain why each error occurs.
5. Provide corrected Python code.
6. Explain the corrections line by line.
7. Suggest improvements where appropriate.

Important rules:

- Do not execute the code.
- Do not claim that you executed or tested it.
- If there are no obvious errors, say so.
- Distinguish confirmed errors from possible issues.
- Do not invent errors.
- Preserve the original purpose of the code.

Structure your response using these headings:

### 1. Code Analysis

### 2. Errors Identified

### 3. Why These Errors Occur

### 4. Corrected Code

### 5. Explanation of Corrections

### 6. Suggestions for Improvement

Python Code:

{code}
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    if not response.text:

        raise ValueError(
            "Gemini returned an empty debugging response."
        )

    return response.text