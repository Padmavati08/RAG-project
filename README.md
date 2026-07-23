# 📚 RAG-Based Educational Assistant

A complete Retrieval-Augmented Generation (RAG) system that answers questions from educational PDFs using local LLMs via Ollama.

---

## 🚀 Overview

This project implements an end-to-end **RAG pipeline** that:

* Loads PDF documents
* Splits them into meaningful chunks
* Converts them into embeddings
* Stores them in a vector database (FAISS)
* Retrieves relevant context for user queries
* Generates accurate answers using a local LLM

---

## 🧠 Key Features

* 📄 PDF-based question answering
* 🔍 Semantic search using FAISS
* 🧩 Context-aware responses (not generic AI answers)
* 🤖 Local LLM support via Ollama (no API cost)
* 🌐 Streamlit web UI
* ⚡ Fast retrieval + response pipeline

---

## 🏗️ Tech Stack

| Component  | Technology                     |
| ---------- | ------------------------------ |
| Language   | Python                         |
| Framework  | LangChain                      |
| Embeddings | Sentence Transformers (MiniLM) |
| Vector DB  | FAISS                          |
| LLM        | Ollama (llama3 / gemma)        |
| UI         | Streamlit                      |
| Input      | PDF Documents                  |

---

## 📂 Project Structure

```
RAG-project/
│
├── data/                   # PDF files
├── src/
│   ├── main.py             # Core RAG pipeline
│   └── app.py              # Streamlit UI
├── vectorstore/            # FAISS index storage
├── requirements.txt
└── README.md
```

---

## ⚙️ How It Works

```
Load PDFs → Split into chunks → Create embeddings
→ Store in FAISS → User query → Retrieve context
→ Generate answer using LLM
```

---

## ▶️ How to Run

### 1. Activate environment

```
venv\Scripts\activate
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run Ollama (make sure it's running)

```
ollama run llama3
```

### 4. Run Streamlit UI

```
streamlit run src/app.py
```

---

## 💬 Example Usage

Ask questions like:
* "Explain difference between e commerce and e business"
  

The system retrieves relevant document chunks and generates a precise answer.

---

## 📅 Development Timeline (3 Weeks)

### Week 1 — Foundation & Data Pipeline

* Project structure setup
* Virtual environment and dependency setup
* PDF loading using LangChain
* Text extraction and preprocessing
* Document chunking using RecursiveCharacterTextSplitter

### Week 2 — Core RAG Implementation

* Embeddings using Sentence Transformers
* FAISS vector database integration
* Similarity search (top-k retrieval)
* Context building from retrieved chunks

### Week 3 — LLM + UI + Optimization

* Ollama integration (llama3 / gemma models)
* Prompt engineering for better answers
* Streamlit UI development
* Fixing errors (paths, embeddings, model timeouts)
* Improving answer quality (concise + structured output)
* Final testing and GitHub deployment

---

## 📊 Final Status

The system is fully functional and capable of:

* Processing PDFs into searchable knowledge
* Retrieving relevant context using vector search
* Generating accurate answers using a local LLM
* Providing an interactive UI via Streamlit


## ✅ Current Status

| Feature                  | Status      |
| ------------------------ | ----------- |
| Project Setup            | ✅ Completed |
| PDF Loading              | ✅ Completed |
| Text Chunking            | ✅ Completed |
| Embeddings               | ✅ Completed |
| Vector Database (FAISS)  | ✅ Completed |
| Retrieval System         | ✅ Completed |
| LLM Integration (Ollama) | ✅ Completed |
| Streamlit UI             | ✅ Completed |

---

## ⚠️ Notes

* Make sure PDFs are inside the `data/` folder
* Ollama must be running before asking questions
* First run may take time due to model loading

---

## 🔮 Future Improvements

* 🔹 Better UI (chat interface, history)
* 🔹 Source citations for answers
* 🔹 Multi-document upload
* 🔹 Deployment (Render / HuggingFace Spaces)
* 🔹 Hybrid search (keyword + semantic)

---

## 📌 Author

**Padmavati Naik**


Give it a star on GitHub ⭐
