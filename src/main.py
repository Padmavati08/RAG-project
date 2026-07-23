import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path

# ------------------ CONFIG ------------------
# ------------------ CONFIG ------------------

BASE_DIR = Path(__file__).resolve().parent.parent  # go to project root

DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "vectorstore/faiss_index"

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# ------------------ LOAD DOCUMENTS ------------------

def load_documents():
    print("📄 Loading PDFs...")

    loader = PyPDFDirectoryLoader(DATA_DIR, glob="**/*.pdf")
    documents = loader.load()

    if not documents:
        raise ValueError("❌ No PDF files found in data/")

    print(f"✅ Loaded {len(documents)} pages")
    return documents


# ------------------ CHUNKING ------------------

def split_documents(documents):
    print("✂️ Splitting documents into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    print(f"✅ Created {len(chunks)} chunks")
    return chunks


# ------------------ VECTOR STORE ------------------

def get_vectorstore(chunks, embeddings):
    index_path = Path(INDEX_DIR)

    if index_path.exists():
        print("📦 Loading existing vector database...")
        return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

    print("🧠 Creating new vector database...")

    index_path.parent.mkdir(parents=True, exist_ok=True)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_DIR)

    print("✅ Vector database saved")
    return vectorstore


# ------------------ RETRIEVAL ------------------

def retrieve_context(vectorstore, query, k=5):
    results = vectorstore.similarity_search(query, k=k)

    # better context control (prevents dumping raw text)
    context = "\n\n".join([doc.page_content[:300] for doc in results])

    return context, results


# ------------------ LLM ------------------

def generate_answer(context, query):
    try:
        llm = ChatOllama(
            model=OLLAMA_MODEL,
            temperature=0
        )

        prompt = ChatPromptTemplate.from_template(
            """You are an expert assistant.

Answer the question using ONLY the context below.

Rules:
- Do NOT copy text directly.
- Give short, clear, structured answers.
- Use bullet points if needed.
- If answer is not present, say "I don't know".

Context:
{context}

Question:
{question}

Answer:"""
        )

        messages = prompt.format_messages(
            context=context,
            question=query
        )

        response = llm.invoke(messages)

        return response.content.strip()

    except Exception as e:
        return f"❌ LLM Error: {str(e)}"


# ------------------ MAIN APP ------------------

def main():
    print("🚀 RAG System Started")

    documents = load_documents()
    chunks = split_documents(documents)

    print("🧠 Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = get_vectorstore(chunks, embeddings)

    while True:
        query = input("\n💬 Ask a question (or type 'exit'): ").strip()

        if query.lower() == "exit":
            print("👋 Exiting...")
            break

        context, results = retrieve_context(vectorstore, query)

        print("\n🔍 Top Retrieved Chunk:\n")
        if results:
            print(results[0].page_content[:500])

        answer = generate_answer(context, query)

        print("\n🤖 Answer:\n")
        print(answer)


# ------------------ ENTRY ------------------

if __name__ == "__main__":
    main()