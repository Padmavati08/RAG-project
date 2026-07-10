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