from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

class DocumentIngestion:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        # Using open-source embedding model (runs locally, no API key needed!)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
            # This is a lightweight, fast model that runs on your machine
        )
        
    def load_document(self, file_path):
        loader = TextLoader(file_path)
        return loader.load()
    
    def chunk_documents(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
            
        return chunks
    
    def create_vector_store(self, chunks):
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        return vectorstore
    
    def ingest_folder(self, folder_path):
        all_chunks = []
        
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            if os.path.isfile(file_path) and filename.endswith('.txt'):
                documents = self.load_document(file_path)
                
                for doc in documents:
                    doc.metadata['source_file'] = filename
                
                chunks = self.chunk_documents(documents)
                all_chunks.extend(chunks)
        
        return self.create_vector_store(all_chunks)

if __name__ == "__main__":
    ingestion = DocumentIngestion()
    vectorstore = ingestion.ingest_folder("./policy_documents")