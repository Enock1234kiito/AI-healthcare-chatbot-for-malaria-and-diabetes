from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstores/db_faiss"

# Create vector database
def create_vector_db():
    try:
        # Ensure the data path exists
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Data path '{DATA_PATH}' does not exist.")

        # Load documents
        loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
        documents = loader.load()

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
        texts = text_splitter.split_documents(documents)

        # Create embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_kwargs={'device': 'cpu'}
        )

        # Create and save vector store
        db = FAISS.from_documents(texts, embeddings)
        db.save_local(DB_FAISS_PATH)

        print("Vector database created successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    create_vector_db()
