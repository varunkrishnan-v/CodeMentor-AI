import faiss
import json
import numpy as np
import os

from sentence_transformers import SentenceTransformer

from document_loader import load_document
from text_chunker import split_text

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

file_path = "C:/Users/varun/OneDrive/Desktop/PROJECTS/RAG Project/data/documents/Python Programming.pdf"

documents = load_document(file_path)
all_chunks = []

for document in documents:

    chunks = split_text(
        document["text"],
        chunk_size=20,
        overlap=10
    )

    for chunk in chunks:

        all_chunks.append({
            "text": chunk,
            "source": document["source"],
            "page": document["page"]
        })
chunk_texts = [chunk["text"] for chunk in all_chunks]
embeddings = model.encode(chunk_texts)
print("Number of chunks:", len(all_chunks))
print("Embedding shape:", embeddings.shape)
embeddings = np.array(embeddings).astype("float32")
faiss.normalize_L2(embeddings)

print("Normalized embeddings shape:", embeddings.shape)

print("Embedding data type:", embeddings.dtype)

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)
index.add(embeddings)
query = ["How do I store information in Python?"]
query_embedding = model.encode(query)
query_embedding = np.array(query_embedding).astype("float32")
faiss.normalize_L2(query_embedding)
scores, indices = index.search(query_embedding, k=2)
print("\nSearch Results:\n")

for i in range(len(indices[0])):

    chunk_index = indices[0][i]

    print("Chunk:", all_chunks[chunk_index]["text"])
    print("Source:", all_chunks[chunk_index]["source"])
    print("Similarity Score:", scores[0][i])
    print("-" * 50)
print("Total vectors stored:", index.ntotal)
print("\nFirst 10 Generated Chunks:\n")

for i, chunk in enumerate(all_chunks[:10], start=1):

    print(f"Chunk {i}:")
    print("Source:", chunk["source"])
    print("Page:", chunk["page"])
    print("Text:", chunk["text"])
    print("-" * 50)


os.makedirs("data/index", exist_ok=True)
faiss.write_index(index, "data/index/faiss_index.bin")

print("FAISS index saved successfully!")

# Save chunks and metadata in JSON format
with open("data/index/chunks.json", "w", encoding="utf-8") as file:

    json.dump(all_chunks, file, ensure_ascii=False, indent=4)

print("Chunks saved successfully!")