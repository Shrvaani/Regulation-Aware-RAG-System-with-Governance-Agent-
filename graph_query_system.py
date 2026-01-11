from graph_retrieval import GraphRetrieval
from governance_agent import GovernanceAgent, GovernanceDecision
from typing import Dict, Any, List

class GraphQuerySystem:
    def __init__(self, graph_db_path="./graph_db"):
        self.graph_retrieval = GraphRetrieval(graph_db_path)
        try:
            self.governance_agent = GovernanceAgent()
        except Exception as e:
            self.governance_agent = None
    
    def answer_question(self, question: str, context: str = "") -> Dict[str, Any]:
        """Answer a question using graph RAG."""
        # Query the graph
        graph_query_result = self.graph_retrieval.query_graph(question)
        
        # Build context from graph relationships
        graph_context = graph_query_result.get("graph_context", "")
        matched_entities = graph_query_result.get("matched_entities", [])
        related_info = graph_query_result.get("related_info", [])
        
        # Format context for LLM
        formatted_context = self._format_graph_context(graph_context, matched_entities, related_info)
        
        # Use governance agent to answer based on graph context
        if self.governance_agent:
            try:
                decision = self.governance_agent.evaluate_action(
                    action=question,
                    policy_context=formatted_context,
                    additional_context=context
                )
                
                # Build response with graph insights
                response = {
                    "question": question,
                    "answer": decision.reason,
                    "decision": self._extract_decision_from_reason(decision.reason),
                    "matched_entities": matched_entities[:5],
                    "graph_paths": self._extract_graph_paths(related_info),
                    "confidence_score": decision.confidence_score,
                    "risk_level": decision.risk_level,
                    "references": decision.references,
                    "suggested_changes": decision.suggested_changes,
                    "alternative_actions": decision.alternative_actions,
                    "graph_context_summary": formatted_context[:500] + "..." if len(formatted_context) > 500 else formatted_context
                }
                
                return response
            except Exception as e:
                # Fallback if LLM evaluation fails
                return self._generate_answer_from_graph_only(question, matched_entities, related_info, formatted_context)
        else:
            # No LLM available, use graph-only reasoning
            return self._generate_answer_from_graph_only(question, matched_entities, related_info, formatted_context)
    
    def _generate_answer_from_graph_only(self, question: str, matched_entities: List[Dict], related_info: List[Dict], graph_context: str) -> Dict[str, Any]:
        """Generate answer using only graph relationships (no LLM)."""
        # Simple rule-based reasoning from graph
        answer = "Based on the knowledge graph:\n\n"
        
        if not matched_entities:
            answer += "No relevant entities found in the knowledge graph for this question."
            return {
                "question": question,
                "answer": answer,
                "decision": "Unknown",
                "matched_entities": [],
                "graph_paths": [],
                "confidence_score": 0.0,
                "risk_level": "Medium",
                "references": [],
                "suggested_changes": ["Ingest relevant documents to build knowledge graph"],
                "alternative_actions": [],
                "graph_context_summary": graph_context[:500] if graph_context else "No context available"
            }
        
        # Analyze relationships to infer answer
        answer_parts = []
        for info in related_info[:3]:
            entity = info.get("entity", {})
            entity_name = entity.get("name", entity.get("id"))
            relationships = info.get("relationships", {})
            
            # Look for key relationships
            for rel in relationships.get("outgoing", []):
                rel_type = rel.get("relationship")
                target_name = rel.get("target_name", rel.get("target"))
                
                if rel_type == "IS_NOT":
                    answer_parts.append(f"{entity_name} is not {target_name}.")
                elif rel_type == "LACKS":
                    answer_parts.append(f"{entity_name} lacks {target_name}.")
                elif rel_type == "REQUIRES":
                    answer_parts.append(f"{entity_name} requires {target_name}.")
            
            for rel in relationships.get("incoming", []):
                rel_type = rel.get("relationship")
                source_name = rel.get("source_name", rel.get("source"))
                
                if rel_type == "REQUIRES":
                    answer_parts.append(f"{source_name} requires {entity_name}.")
        
        if answer_parts:
            answer += "\n".join(answer_parts[:5])
            answer += "\n\nRelationship analysis from knowledge graph."
        else:
            answer += "Found relevant entities but no clear relationships to answer the question."
        
        # Try to infer decision from relationships
        decision = "Unknown"
        if any("IS_NOT" in str(info) for info in related_info):
            decision = "Not a Crime"
        elif any("REQUIRES" in str(info) for info in related_info):
            decision = "Conditional"
        
        return {
            "question": question,
            "answer": answer,
            "decision": decision,
            "matched_entities": matched_entities[:5],
            "graph_paths": self._extract_graph_paths(related_info),
            "confidence_score": 0.5 if matched_entities else 0.0,
            "risk_level": "Medium",
            "references": [f"Graph entity: {e.get('name', e.get('id'))}" for e in matched_entities[:3]],
            "suggested_changes": [],
            "alternative_actions": [],
            "graph_context_summary": graph_context[:500] if graph_context else "No context available"
        }
    
    def _format_graph_context(self, graph_context: str, entities: List[Dict], related_info: List[Dict]) -> str:
        """Format graph context for LLM prompt."""
        context = "KNOWLEDGE GRAPH CONTEXT:\n\n"
        
        if graph_context:
            context += graph_context
        else:
            context += "No relevant entities found in graph.\n\n"
        
        # Add entity information
        if entities:
            context += "\nRELEVANT ENTITIES FROM GRAPH:\n"
            for entity in entities[:5]:
                entity_name = entity.get("name", entity.get("id"))
                entity_type = entity.get("type", "Unknown")
                props = entity.get("properties", {})
                
                context += f"- {entity_type}: {entity_name}\n"
                if "description" in props:
                    context += f"  Description: {props['description']}\n"
                if "text" in props:
                    context += f"  Text: {props['text']}\n"
                context += "\n"
        
        # Add relationship chains
        if related_info:
            context += "\nRELATIONSHIP CHAINS:\n"
            for info in related_info[:3]:
                entity = info.get("entity", {})
                entity_name = entity.get("name", entity.get("id"))
                relationships = info.get("relationships", {})
                
                # Show key relationships
                outgoing = relationships.get("outgoing", [])
                for rel in outgoing[:3]:
                    if rel.get("relationship") in ["REQUIRES", "LACKS", "IS_NOT"]:
                        target = rel.get("target_name", rel.get("target"))
                        rel_type = rel.get("relationship")
                        context += f"- {entity_name} {rel_type} {target}\n"
                context += "\n"
        
        return context
    
    def _extract_decision_from_reason(self, reason: str) -> str:
        """Extract yes/no decision from reason text."""
        reason_lower = reason.lower()
        
        if any(word in reason_lower for word in ["not a crime", "not crime", "not illegal", "not violate", "no, it is not", "no, this is not"]):
            return "Not a Crime"
        elif any(word in reason_lower for word in ["is a crime", "is illegal", "is violate", "yes, it is", "yes, this is"]):
            return "Is a Crime"
        elif any(word in reason_lower for word in ["conditional", "depends", "may be"]):
            return "Conditional"
        else:
            return "Unknown"
    
    def _extract_graph_paths(self, related_info: List[Dict]) -> List[Dict]:
        """Extract important graph paths from related info."""
        paths = []
        
        for info in related_info[:3]:
            entity = info.get("entity", {})
            entity_name = entity.get("name", entity.get("id"))
            
            relationships = info.get("relationships", {})
            
            # Find key paths like: Accidental → LACKS → Intent → REQUIRED_BY → Theft
            for rel in relationships.get("outgoing", []):
                if rel.get("relationship") in ["REQUIRES", "LACKS", "IS_NOT"]:
                    paths.append({
                        "source": entity_name,
                        "relationship": rel.get("relationship"),
                        "target": rel.get("target_name", rel.get("target"))
                    })
            
            for rel in relationships.get("incoming", []):
                if rel.get("relationship") in ["REQUIRES", "LACKS", "IS_NOT"]:
                    paths.append({
                        "source": rel.get("source_name", rel.get("source")),
                        "relationship": rel.get("relationship"),
                        "target": entity_name
                    })
        
        return paths[:10]  # Limit to 10 paths

