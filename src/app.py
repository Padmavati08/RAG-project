import streamlit as st
from main import load_documents, split_documents, get_vectorstore, retrieve_context, generate_answer
from langchain_community.embeddings import HuggingFaceEmbeddings

st.set_page_config(page_title="RAG QA System", layout="wide")

st.title("📚 RAG Document QA System")

@st.cache_resource
def setup():
    documents = load_documents()
    chunks = split_documents(documents)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = get_vectorstore(chunks, embeddings)
    return vectorstore

# ✅ FIXED INDENTATION
with st.spinner("Setting up RAG system... please wait ⏳"):
    vectorstore = setup()

query = st.text_input("Ask a question:")

if query:
    context, results = retrieve_context(vectorstore, query)

    st.subheader("🔍 Retrieved Context")
    if results:
        st.write(results[0].page_content[:500])

    answer = generate_answer(context, query)

    st.subheader("🤖 Answer")
    st.write(answer)

st.success("✅ System ready!")