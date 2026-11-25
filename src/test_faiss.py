import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "Python variables store information.",
    "Variables are used to save data in Python.",
    "Neural networks are used in deep learning."
]

embeddings = model.encode(documents)
embeddings = np.array(embeddings).astype("float32")
faiss.normalize_L2(embeddings)
dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)
index.add(embeddings)
query = ["How do I save information in Python?"]
query_embedding = model.encode(query)
query_embedding = np.array(query_embedding).astype("float32")
faiss.normalize_L2(query_embedding)
scores, indices = index.search(query_embedding, k=2)

print("Search Results:\n")

for i in range(len(indices[0])):
    print("Document:", documents[indices[0][i]])
    print("Similarity Score:", scores[0][i])
    print()