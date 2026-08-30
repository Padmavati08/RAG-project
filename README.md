# 🎓 RAG Educational Assistant & Evaluation Suite

An end-to-end **Retrieval-Augmented Generation (RAG)** educational platform engineered with a vector database, cosine-similarity retrieval pipeline, transparent source grounding, and an automated quality evaluation framework.

## 📌 System Architecture
┌─────────────────────────────┐
                          │       Knowledge Corpus      │
                           │  (Document Ingestion / DB)  │
                           └──────────────┬──────────────┘
                                          │ Chunking & Embeddings
                                          ▼┌──────────────────┐ ┌─────────────────────────┐
│ User Query │──────────────►│ Vector Database │
└──────────────────┘ │ (Cosine Similarity) │
└──────────┬──────────────┘
│ Top-K Chunks
▼
┌────────────────────────────────────────────────────────────┐
│ RAG Synthesis Engine │
│ - Context Assembly & Grounded Prompt Construction │
│ - LLM Generation (Gemini API / Grounded Synthesizer) │
└─────────────────────────────┬──────────────────────────────┘
│
▼
┌────────────────────────────────────────────────────────────┐
│ Grounded Response │
│ - Verified Answer Content │
│ - Exact Source Citations & Chunk IDs │
│ - Match Confidence Scores │
└─────────────────────────────┬──────────────────────────────┘
│
▼
┌────────────────────────────────────────────────────────────┐
│ Automated Evaluation Framework │
│ - Faithfulness Score (Grounding Validation) │
│ - Context Precision (Noise Ratio) │
│ - Answer Relevance (Query Alignment) │
└────────────────────────────────────────────────────────────┘


## 🧠 Key Features & Functional Modules

### 1. 🔍 Vector Database & Retrieval Engine
- **Pre-Indexed Knowledge Base:** 40+ structured domain chunks covering e-commerce architectures, IT infrastructure, B2B procurement workflows, and digital business models.
- **Vector Space Modeling:** High-dimensional vector representations computed for user queries and knowledge chunks.
- **Cosine Similarity Search:** Real-time semantic similarity ranking (0.0 to 1.0) with configurable relevance thresholds and top-k filtering.

### 2. 🤖 Grounded Educational Assistant
**Zero-Hallucination Generation:** Ensures the language model generates answers strictly constrained by retrieved source context.
- **Transparent Source Citations:** Every answer displays chunk references, document titles, similarity percentages, and text snippets used during synthesis.
- **Interactive Question Workbench:** Pre-loaded domain queries (e-commerce vs e-business, Tata Steel SAP procurement, EDI standards, supply chain models).

### 3. 📊 Automated Quality Evaluation Suite
Evaluates and benchmarks RAG performance across industry-standard RAGAS metrics:
- **Faithfulness Score:** Checks factual alignment between the generated response and retrieved context chunks.
- **Context Precision / Relevance:** Measures noise reduction and the proportion of useful information retrieved.
- **Answer Relevance:** Validates how directly the answer addresses the user's specific prompt.
- **Comparative Benchmarks:** Side-by-side comparison of Grounded RAG vs. Non-RAG baseline outputs to demonstrate hallucination reduction.

### 4. 📂 Dynamic Document Ingestion
- Real-time ingestion interface supporting `.pdf` and `.txt` lecture notes for chunking, embedding, and indexing on the fly.

---

## 🏗️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | React 18, TypeScript, Tailwind CSS, Lucide Icons, Motion |
| **Backend** | Node.js, Express, TSX |
| **Vector Engine** | In-Memory Vector Store, Cosine Similarity, Vector Space Models |
| **LLM & Synthesis** | Google Gemini API / Deterministic Grounded Synthesizer |
| **Evaluation** | Automated RAGAS-inspired Metric Engine |

---

## 📁 Project Directory Structure
RAG-project/
├── src/
│ ├── components/ # UI Components (Q&A Assistant, Vector Explorer, Evaluation Suite, Pipeline Visualizer)
│ ├── data/ # Vector Knowledge Chunks & Evaluation Benchmark Datasets
│ ├── types.ts # TypeScript Types & Interfaces
│ ├── App.tsx # Main Application Dashboard
│ └── main.tsx # React Root Entrypoint
├── server.ts # Express Backend & RAG Endpoints
├── package.json # Dependencies & Scripts
├── tsconfig.json # TypeScript Compiler Configuration
├── vite.config.ts # Vite Build Configuration
└── README.md # Project Documentation
