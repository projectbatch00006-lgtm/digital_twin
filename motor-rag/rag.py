import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path


embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


client = chromadb.PersistentClient(
    path=str(Path(__file__).resolve().parent / "chroma_db")
)


try:
    collection = client.get_collection(name="motor_procedures")
except Exception as error:
    raise RuntimeError(
        "Chroma collection 'motor_procedures' was not found. "
        "Run 'python ingest.py' from the motor-rag folder first."
    ) from error


def retrieve_procedure(fault_type, top_k=2):

    if not fault_type or not fault_type.strip():
        raise ValueError("A motor fault description is required.")

    count = collection.count()
    if count == 0:
        raise RuntimeError(
            "The motor procedure collection is empty. Run 'python ingest.py' first."
        )

    query_embedding = embedding_model.encode(
        fault_type
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, count)
    )

    return results
