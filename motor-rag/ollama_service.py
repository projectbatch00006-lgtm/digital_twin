from ollama import chat

def generate_instruction(
    fault_type,
    retrieved_context
):

    prompt = f"""
You are an industrial motor maintenance
assistance system.

The ML and diagnostic pipeline identified the following
likely fault condition:

FAULT:
{fault_type}

RETRIEVED KNOWLEDGE:
{retrieved_context}

Based ONLY on the retrieved knowledge:

1. Explain the detected fault condition.
2. Provide the relevant inspection procedure.
3. Provide the recovery/maintenance steps.
4. Clearly list safety warnings.
5. Do not invent procedures.
6. Do not recommend bypassing protection systems.
7. Do not instruct anyone to work on energized equipment.
8. Escalate to qualified maintenance personnel when required.

Format the response clearly.
"""

    response = chat(
        model="mistral:latest",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]
