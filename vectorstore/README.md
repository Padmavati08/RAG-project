# 🗄️ Vector Store Directory (`/vectorstore`)

This directory manages serialized vector indices, embeddings metadata, and pre-computed similarity stores for educational document retrieval.

---

## 📂 Subdirectories & Artifacts

| Path | Format | Description |
| :--- | :--- | :--- |
| `faiss_index/` | FAISS Index Directory | Contains serialized FAISS vector index files for fast nearest-neighbor lookups. |
| `faiss_index/index.faiss` | Binary Vector Index | Serialized L2/Inner-Product multidimensional embedding vectors computed from chunked educational documents. |
| `faiss_index/index.pkl` | Pickled Metadata | Serialized document metadata, chunk text mappings, source document pointers, and token frequencies. |

---

## 🔍 Retrieval & Indexing Specifications

* **Chunk Size**: `500 characters`
* **Chunk Overlap**: `50 characters`
* **Embedding Model**: `Sentence Transformers (MiniLM / 768-dim dense embeddings)`
* **Similarity Metrics**: `Cosine Similarity & BM25 Hybrid Scoring`
* **Index Type**: `In-Memory Dense Vector Store / FAISS Index`
* **Top-K Parameter**: `5 chunks per query`
