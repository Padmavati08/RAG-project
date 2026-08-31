# 📁 Data Directory (`/data`)

This directory houses the raw educational source documents, lecture notes, and extracted structured artifacts used by the **Retrieval-Augmented Generation (RAG)** pipeline.

---

## 📑 Contents

| File | Type | Description |
| :--- | :--- | :--- |
| `notes.pdf` | PDF Document | Primary 15-page course document: *"Week 1: Introduction to E-Business & E-Commerce"*. Contains core definitions, value chain integrations, Tata Steel SAP e-procurement workflows, and ITC e-Choupal case studies. |
| `extracted_notes.json` | JSON Artifact | Serialized page-by-page text extractions and preprocessed metadata parsed for rapid vector indexing and testing. |

---

## ⚙️ Ingestion & Processing Workflow

```
PDF Document (data/notes.pdf) 
       │
       ▼
Text Extraction (pdf-parse / buffer)
       │
       ▼
Recursive Character Chunking (500 chars, 50 chars overlap)
       │
       ▼
Vector Database Indexing (In-Memory Dense Store + TF-IDF)
```

---

## 💡 Adding New Knowledge Documents

1. **Via Web Interface**: Click **"Add Knowledge PDF"** in the top navigation bar to upload any `.pdf` or paste text directly.
2. **Via Filesystem**: Place new `.pdf` files in this `/data` folder. The RAG server will automatically detect and ingest them during startup or rebuilds.
