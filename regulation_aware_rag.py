from retrieval_system import RetrievalSystem
from governance_agent import GovernanceAgent
from typing import Dict

class RegulationAwareRAG:
    def __init__(self, vector_db_path="./chroma_db"):
        self.retrieval_system = RetrievalSystem(vector_db_path)
        self.governance_agent = GovernanceAgent()
    
    def process_action(self, action: str, context: str = "", top_k: int = 5) -> Dict:
        # Retrieve relevant policies
        retrieved_docs = self.retrieval_system.retrieve_relevant_policies(action, k=top_k)
        
        # Format context
        policy_context = self.retrieval_system.format_context_for_llm(retrieved_docs)
        
        # Evaluate with governance agent
        decision = self.governance_agent.evaluate_action(action, policy_context, context)
        
        # Compile response
        response = {
            "action": action,
            "decision": decision.decision,
            "reason": decision.reason,
            "suggested_changes": decision.suggested_changes,
            "references": decision.references,
            "risk_level": decision.risk_level,
            "alternative_actions": decision.alternative_actions,
            "confidence_score": decision.confidence_score,
            "retrieved_policy_count": len(retrieved_docs),
            "policy_sources": list(set([doc['source'] for doc in retrieved_docs])) if retrieved_docs else []
        }
        
        return response