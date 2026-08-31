# 💻 Source Code Directory (`/src`)

This directory contains the client-side **React + TypeScript** web application frontend and interactive user interfaces for the Educational RAG system.

---

## 📂 File Structure

| File | Purpose | Key Responsibilities |
| :--- | :--- | :--- |
| `App.tsx` | Main Application Component | Orchestrates the 5 primary tabs (Q&A Assistant, RAG Pipeline Architecture, Vector Database Explorer, Response Quality Evaluation Suite, and Knowledge Documents Manager). Includes real-time metric scorecards, baseline comparison mode, and chunk inspector. |
| `main.tsx` | React Entry Point | Mounts the root React DOM node and applies strict mode with application styling. |
| `index.css` | Global Tailwind CSS Entry | Implements Tailwind utility directives and custom font pairings (*Outfit* for headings, *Plus Jakarta Sans* for UI body, and *JetBrains Mono* for vector code previews). |

---

## 🌟 Key User Interface Features

1. **Q&A Assistant & Grounding**: Interactive search bar with starter curriculum prompts, live streaming answer cards, and Top-K retrieved vector chunk cards with similarity scores.
2. **Side-by-Side Baseline Comparison**: Shows vanilla LLM output (hallucination-prone) alongside Vector-Augmented grounded output.
3. **RAG Pipeline Architecture**: Live visual execution trace showing duration and operations across Tokenization, Vector DB Search, Prompt Assembly, and LLM Inference.
4. **Vector Database Explorer**: Interactive table displaying chunk IDs, character offsets, token counts, keywords, and 8D dense vector representations.
5. **Quality Evaluation Suite**: Real-time evaluation of **Faithfulness**, **Context Precision**, and **Answer Relevance**, plus a 1-click automated benchmark runner across 5 standardized test queries.
