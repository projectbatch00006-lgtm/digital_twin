"""Build the local Chroma knowledge base from approved motor documents."""

import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "motor_procedures"


def clean_text(text):
    return re.sub(r"\s+", " ", text or "").strip()


def chunk_text(text, chunk_size=1800, overlap=250):
    text = clean_text(text)
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(".", start, end)
            if boundary > start + chunk_size // 2:
                end = boundary + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def manufacturer_from_name(name):
    lower = name.lower()
    for key, value in (("weg", "WEG"), ("nidec", "Nidec")):
        if key in lower:
            return value
    return "Unknown"


def load_pdf_documents(path):
    reader = PdfReader(str(path))
    records = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            records.append({
                "text": text,
                "source": path.name,
                "page": page_number,
                "manufacturer": manufacturer_from_name(path.name),
                "document_type": "approved_manual"
            })
    return records


def load_docx_documents(path):
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml")
    root = ElementTree.fromstring(xml)
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    paragraphs = []
    for paragraph in root.iter(namespace + "p"):
        words = [node.text or "" for node in paragraph.iter(namespace + "t")]
        value = clean_text("".join(words))
        if value:
            paragraphs.append(value)
    return [{
        "text": " ".join(paragraphs),
        "source": path.name,
        "page": None,
        "manufacturer": "Unknown",
        "document_type": "approved_guide"
    }]


def load_structured_procedures(path):
    with path.open("r", encoding="utf-8") as file:
        procedures = json.load(file)
    records = []
    for item in procedures:
        records.append({
            "text": (
                f"Fault: {item['fault']}\nTitle: {item['title']}\n"
                f"Procedure: {' '.join(item['procedure'])}\n"
                f"Warnings: {' '.join(item['warnings'])}"
            ),
            "source": path.name,
            "page": None,
            "manufacturer": "General",
            "document_type": "approved_procedure",
            "fault": item["fault"]
        })
    return records


def main():
    print("Loading embedding model...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    records = load_structured_procedures(DATA_DIR / "procedures.json")
    document_dirs = [DATA_DIR]
    nested_documents_dir = DATA_DIR / "documents"
    if nested_documents_dir.exists():
        document_dirs.append(nested_documents_dir)
    pdf_paths = sorted({path for directory in document_dirs for path in directory.glob("*.pdf")})
    docx_paths = sorted({path for directory in document_dirs for path in directory.glob("*.docx")})
    for path in pdf_paths:
        records.extend(load_pdf_documents(path))
    for path in docx_paths:
        records.extend(load_docx_documents(path))

    documents, metadatas, ids = [], [], []
    for record_index, record in enumerate(records):
        for chunk_index, chunk in enumerate(chunk_text(record["text"])):
            documents.append(chunk)
            metadata = {
                "source": record["source"],
                "page": record["page"] if record["page"] is not None else -1,
                "manufacturer": record["manufacturer"],
                "document_type": record["document_type"]
            }
            if "fault" in record:
                metadata["fault"] = record["fault"]
            metadatas.append(metadata)
            ids.append(f"{record_index}-{chunk_index}")

    if not documents:
        raise RuntimeError("No readable documents were found for indexing.")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(name=COLLECTION_NAME)
    print(f"Embedding {len(documents)} document chunks...")
    embeddings = embedding_model.encode(documents, show_progress_bar=True).tolist()
    collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
    print("Knowledge base indexed successfully.")
    print("Total chunks:", collection.count())
    print("Sources:", len(set(item["source"] for item in metadatas)))


if __name__ == "__main__":
    main()
