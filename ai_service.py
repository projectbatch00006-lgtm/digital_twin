"""Shared Groq client used for fault detection and RAG generation."""

import os

from groq import Groq


GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


def generate_ai_response(prompt, system_prompt=None):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Configure it before running the application."
        )

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.1,
        max_tokens=1200,
        messages=[
            {
                "role": "system",
                "content": system_prompt or (
                    "You are an industrial motor maintenance assistant. "
                    "Use only the evidence supplied by the application. "
                    "Do not claim a fault is confirmed without inspection."
                )
            },
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content
