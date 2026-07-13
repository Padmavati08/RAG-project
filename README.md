RAG-Based Educational Assistant
A Python project that uses Ollama, LLMs, and a vector database to answer questions from educational documents.

# Overview
This project follows a RAG pipeline for document-based question answering.

## Tech Stack

| Component | Tech             |
| Language  | Python           |
| LLM       | Ollama           |
| Method    | RAG              |
| Storage   | Vector Database  |
| Input     | PDF/Text Document|

## Progress
Project structure        | Completed   |
dependency setup         | Completed   |
PDF loading              | Completed   |
text extraction          | Completed   |
document chunking        | Completed   |
Embeddings               | In progress |
vector db integration    | In progress |
LLM response generation  | In progress |
deployment               | In progress |

## Flow

flowchart TD
A[Load documents] --> B[Extract text]
B --> C[Chunk text]
C --> D[Create embeddings]
D --> E[Store in vector database]
E --> F[Ask question]
F --> G[Retrieve relevant chunks]
G --> H[Generate answer with Ollama]
```

## Folder Structure

```bash
RAG-project/
├── data/
├── src/
├── reports/
├── deployment/
├── main.py
├── requirements.txt
└── README.md
```

## Current Status

Work in progress. Core preprocessing is done, and the remaining RAG and deployment parts are pending.

# Weekly Report

## Project
RAG-Based Educational Assistant

## Week 1
Completed 2 commits:
- Initial project structure setup.
- Virtual environment and dependencies setup.

## Week 2
Completed 2 commits:
- PDF loading and text extraction.
- Document chunking.

## Current status
Rest of the work is in progress.

## Next step
Add embeddings, vector database integration, Ollama connection, and deployment files.
