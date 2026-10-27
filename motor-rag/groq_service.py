"""Groq-backed generation for approved maintenance procedures."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_service import generate_ai_response


def generate_instruction(fault_type, retrieved_context):
    prompt = f"""
The ML pipeline identified this likely motor condition:

{fault_type}

Use ONLY the following approved maintenance knowledge:

{retrieved_context}

Return:
1. Likely fault interpretation
2. Inspection procedure
3. Recovery or maintenance steps
4. Safety warnings
5. Required escalation

Do not invent procedures. Do not bypass protection systems. Do not instruct
anyone to work on energized equipment. State that qualified personnel must
confirm the fault.
"""
    return generate_ai_response(prompt)
