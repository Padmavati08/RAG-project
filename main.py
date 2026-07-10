print("RAG project setup")
from langchain_community.document_loaders import PyPDFLoader

# Path to your PDF
file_path = "data/notes.pdf"

# Load PDF
loader = PyPDFLoader(file_path)
documents = loader.load()

# Print number of pages
print("Total pages:", len(documents))

# Print first page content
print("\n--- First Page Content ---\n")
print(documents[0].page_content)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Create text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# Split documents into chunks
chunks = text_splitter.split_documents(documents)

# Print info
print("Total chunks:", len(chunks))

print("\n--- First Chunk ---\n")
print(chunks[0].page_content)