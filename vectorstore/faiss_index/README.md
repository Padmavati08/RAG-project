# ⚡ FAISS Vector Index Directory (`/vectorstore/faiss_index`)

This directory stores the specific binary FAISS (Facebook AI Similarity Search) index files and metadata mappings generated from educational course documents.

---

## 📄 Index Files

1. **`index.faiss`**:
   - Binary index storing dense float32 vector embeddings.
   - Enables sub-millisecond approximate nearest neighbor (ANN) search across large collections of educational text chunks.

2. **`index.pkl`**:
   - Python pickle object containing the document chunk payload, character index offsets, chunk titles, and doc IDs corresponding to each vector ID in `index.faiss`.

---

## 🔄 Regeneration

To regenerate or update this index when new documents are added, run the vector indexing pipeline from `server.ts` or via the interactive **Vector Database Explorer** in the web application.
