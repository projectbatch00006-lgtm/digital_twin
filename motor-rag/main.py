"""Command-line entry point for the motor maintenance RAG assistant."""

import sys
from pathlib import Path

# Allow this file to be run either from motor-rag or from the project root.
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from rag import retrieve_procedure
from ollama_service import generate_instruction


def main():
    fault = input("Enter motor fault or sensor condition: ").strip()
    if not fault:
        print("No fault description was entered.")
        return 1

    try:
        results = retrieve_procedure(fault, top_k=3)
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
    except Exception as error:
        print(f"Could not retrieve maintenance procedures: {error}")
        return 1

    if not documents:
        print("No approved maintenance procedure was found for that condition.")
        return 1

    context_parts = []
    for index, document in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) else {}
        source = metadata.get("source", "Unknown source")
        page = metadata.get("page", -1)
        location = f"{source}, page {page}" if page != -1 else source
        context_parts.append(f"SOURCE: {location}\n{document}")
    context = "\n\n".join(context_parts)

    try:
        answer = generate_instruction(fault, context)
    except Exception as error:
        print(f"Could not generate the Mistral response: {error}")
        print("Confirm that Ollama is running and mistral:latest is installed.")
        return 1

    print("\n" + "=" * 60)
    print("INDUSTRIAL MOTOR RECOVERY ASSISTANT")
    print("=" * 60)
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
