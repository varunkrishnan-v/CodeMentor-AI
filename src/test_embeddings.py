from sentence_transformers import SentenceTransformer

# Load the embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Our sample sentences
sentences = [
    "Python variables store information.",
    "Variables are used to save data in Python.",
    "Neural networks are used in deep learning."
]

# Convert sentences into embeddings
embeddings = model.encode(sentences)

# Display the results
print("Embedding shape:", embeddings.shape)

print("\nFirst sentence embedding:")
print(embeddings[0])

print("\nFirst 10 numbers:")
print(embeddings[0][:10])

from sklearn.metrics.pairwise import cosine_similarity

# Calculate similarity between sentences
similarity = cosine_similarity(embeddings)

print("\nSimilarity Matrix:")
print(similarity)