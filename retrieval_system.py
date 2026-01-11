from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List, Dict

class RetrievalSystem:
    def __init__(self, persist_directory="./chroma_db"):
        # Same embedding model as ingestion
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
    
    def retrieve_relevant_policies(self, query: str, k: int = 5) -> List[Dict]:
        try:
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k
            )
            
            if not results or len(results) == 0:
                return []
            
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'relevance_score': float(score),
                    'source': doc.metadata.get('source_file', 'Unknown')
                })
            
            return formatted_results
        except Exception:
            return []
    
    def format_context_for_llm(self, retrieved_docs: List[Dict]) -> str:
        context = "RELEVANT POLICY SECTIONS:\n\n"
        
        for i, doc in enumerate(retrieved_docs, 1):
            context += f"[Section {i}] Source: {doc['source']}\n"
            context += f"Content: {doc['content']}\n\n"
        
        return context