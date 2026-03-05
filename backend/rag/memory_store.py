import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

dimension = 384
index_file = "backend/rag/memory.index"

memory = []

# Load existing index OR create new one
if os.path.exists(index_file):
    print("Loading existing FAISS memory index")
    index = faiss.read_index(index_file)
else:
    print("Creating new FAISS memory index")
    index = faiss.IndexFlatL2(dimension)


def store_memory(text):

    embedding = model.encode([text])

    embedding = np.array(embedding).astype("float32")

    index.add(embedding)

    memory.append(text)

    # Save index to disk
    faiss.write_index(index, index_file)


def retrieve_memory(query, k=3):

    if len(memory) == 0:
        return ""

    query_embedding = model.encode([query])

    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, k)

    results = []

    for idx in indices[0]:
        if idx < len(memory):
            results.append(memory[idx])

    return "\n".join(results)