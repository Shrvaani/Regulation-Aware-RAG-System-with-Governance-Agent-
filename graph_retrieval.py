import networkx as nx
import json
import os
import pickle
from typing import List, Dict, Any, Optional, Set
import re

class GraphRetrieval:
    def __init__(self, graph_db_path="./graph_db"):
        self.graph_db_path = graph_db_path
        self.graph = nx.DiGraph()
        self.load_graph()
    
    def load_graph(self):
        """Load graph from disk."""
        graph_file = os.path.join(self.graph_db_path, "knowledge_graph.pkl")
        
        if os.path.exists(graph_file):
            try:
                with open(graph_file, "rb") as f:
                    loaded_graph = pickle.load(f)
                    # Convert lists back to sets if needed
                    for node in loaded_graph.nodes():
                        props = loaded_graph.nodes[node]
                        if "source_files" in props and isinstance(props["source_files"], list):
                            props["source_files"] = set(props["source_files"])
                    
                    for source, target in loaded_graph.edges():
                        props = loaded_graph.edges[source, target]
                        if "source_files" in props and isinstance(props["source_files"], list):
                            props["source_files"] = set(props["source_files"])
                    
                    self.graph = loaded_graph
            except Exception:
                self.graph = nx.DiGraph()
        else:
            self.graph = nx.DiGraph()
    
    def find_related_entities(self, entity_query: str, max_results: int = 10) -> List[Dict]:
        """Find entities matching the query by name or description."""
        matches = []
        query_lower = entity_query.lower()
        
        for node, props in self.graph.nodes(data=True):
            node_name = props.get("entity_name", "").lower()
            node_id = str(node).lower()
            description = props.get("description", "").lower()
            
            # Check if query matches node
            if (query_lower in node_name or 
                query_lower in node_id or 
                query_lower in description or
                any(query_lower in str(v).lower() for v in props.values() if isinstance(v, str))):
                matches.append({
                    "id": node,
                    "type": props.get("entity_type", "Unknown"),
                    "name": props.get("entity_name", node),
                    "properties": {k: (list(v) if isinstance(v, set) else v) 
                                 for k, v in props.items() if k not in ["entity_type", "entity_name"]}
                })
        
        return matches[:max_results]
    
    def get_entity_relationships(self, entity_id: str, relationship_types: Optional[List[str]] = None) -> Dict[str, List[Dict]]:
        """Get all relationships for an entity."""
        if not self.graph.has_node(entity_id):
            return {"outgoing": [], "incoming": []}
        
        outgoing = []
        incoming = []
        
        # Outgoing edges (entity → other)
        for target in self.graph.successors(entity_id):
            edge_data = self.graph.edges[entity_id, target]
            rel_type = edge_data.get("relationship_type", "RELATES_TO")
            
            if not relationship_types or rel_type in relationship_types:
                outgoing.append({
                    "target": target,
                    "relationship": rel_type,
                    "target_name": self.graph.nodes[target].get("entity_name", target),
                    "target_type": self.graph.nodes[target].get("entity_type", "Unknown"),
                    "properties": {k: (list(v) if isinstance(v, set) else v) 
                                 for k, v in edge_data.items() if k != "relationship_type"}
                })
        
        # Incoming edges (other → entity)
        for source in self.graph.predecessors(entity_id):
            edge_data = self.graph.edges[source, entity_id]
            rel_type = edge_data.get("relationship_type", "RELATES_TO")
            
            if not relationship_types or rel_type in relationship_types:
                incoming.append({
                    "source": source,
                    "relationship": rel_type,
                    "source_name": self.graph.nodes[source].get("entity_name", source),
                    "source_type": self.graph.nodes[source].get("entity_type", "Unknown"),
                    "properties": {k: (list(v) if isinstance(v, set) else v) 
                                 for k, v in edge_data.items() if k != "relationship_type"}
                })
        
        return {"outgoing": outgoing, "incoming": incoming}
    
    def find_paths(self, source_entity: str, target_entity: str, max_length: int = 5) -> List[List[str]]:
        """Find paths between two entities."""
        if not self.graph.has_node(source_entity) or not self.graph.has_node(target_entity):
            return []
        
        try:
            paths = list(nx.all_simple_paths(self.graph, source_entity, target_entity, cutoff=max_length))
            return paths[:10]  # Limit to first 10 paths
        except Exception:
            return []
    
    def traverse_from_entity(self, entity_id: str, relationship_types: Optional[List[str]] = None, depth: int = 2) -> Dict[str, Any]:
        """Traverse graph from an entity to find related entities."""
        if not self.graph.has_node(entity_id):
            return {"entity": entity_id, "found": False, "related": []}
        
        visited = set([entity_id])
        related = []
        queue = [(entity_id, 0)]  # (node, depth)
        
        while queue:
            current, current_depth = queue.pop(0)
            
            if current_depth >= depth:
                continue
            
            # Get all neighbors
            for neighbor in self.graph.successors(current):
                if neighbor not in visited:
                    edge_data = self.graph.edges[current, neighbor]
                    rel_type = edge_data.get("relationship_type", "RELATES_TO")
                    
                    if not relationship_types or rel_type in relationship_types:
                        neighbor_props = self.graph.nodes[neighbor]
                        related.append({
                            "entity_id": neighbor,
                            "entity_name": neighbor_props.get("entity_name", neighbor),
                            "entity_type": neighbor_props.get("entity_type", "Unknown"),
                            "relationship": rel_type,
                            "depth": current_depth + 1,
                            "properties": {k: (list(v) if isinstance(v, set) else v) 
                                         for k, v in neighbor_props.items() 
                                         if k not in ["entity_type", "entity_name"]}
                        })
                        visited.add(neighbor)
                        queue.append((neighbor, current_depth + 1))
            
            # Also check incoming edges
            for neighbor in self.graph.predecessors(current):
                if neighbor not in visited:
                    edge_data = self.graph.edges[neighbor, current]
                    rel_type = edge_data.get("relationship_type", "RELATES_TO")
                    
                    if not relationship_types or rel_type in relationship_types:
                        neighbor_props = self.graph.nodes[neighbor]
                        related.append({
                            "entity_id": neighbor,
                            "entity_name": neighbor_props.get("entity_name", neighbor),
                            "entity_type": neighbor_props.get("entity_type", "Unknown"),
                            "relationship": f"reverse_{rel_type}",
                            "depth": current_depth + 1,
                            "properties": {k: (list(v) if isinstance(v, set) else v) 
                                         for k, v in neighbor_props.items() 
                                         if k not in ["entity_type", "entity_name"]}
                        })
                        visited.add(neighbor)
                        queue.append((neighbor, current_depth + 1))
        
        return {
            "entity": entity_id,
            "entity_name": self.graph.nodes[entity_id].get("entity_name", entity_id),
            "found": True,
            "related": related
        }
    
    def query_graph(self, question: str) -> Dict[str, Any]:
        """Query the graph based on a natural language question."""
        question_lower = question.lower()
        
        # Extract key concepts from question
        concepts = self._extract_concepts(question)
        
        # Find matching entities
        matched_entities = []
        for concept in concepts:
            matches = self.find_related_entities(concept, max_results=5)
            matched_entities.extend(matches)
        
        # Remove duplicates
        seen_ids = set()
        unique_entities = []
        for entity in matched_entities:
            if entity["id"] not in seen_ids:
                seen_ids.add(entity["id"])
                unique_entities.append(entity)
        
        # For each matched entity, get relationships
        related_info = []
        for entity in unique_entities[:5]:  # Limit to top 5
            entity_id = entity["id"]
            relationships = self.get_entity_relationships(entity_id)
            
            # Traverse graph to find related entities
            traversal = self.traverse_from_entity(entity_id, depth=2)
            
            related_info.append({
                "entity": entity,
                "relationships": relationships,
                "traversal": traversal
            })
        
        # Build context from graph
        context = self._build_context_from_graph(related_info, question)
        
        return {
            "question": question,
            "matched_entities": unique_entities[:10],
            "graph_context": context,
            "related_info": related_info
        }
    
    def _extract_concepts(self, question: str) -> List[str]:
        """Extract key concepts/entities from question."""
        concepts = []
        question_lower = question.lower()
        
        # Common legal/policy keywords
        keywords = {
            "theft", "steal", "mobile", "phone", "property",
            "accident", "accidental", "intentional", "intent",
            "crime", "criminal", "illegal", "illegally",
            "require", "required", "must", "should",
            "approval", "consent", "permission",
            "data", "storage", "external", "server",
            "dpo", "data protection", "encryption"
        }
        
        # Find matching keywords
        for keyword in keywords:
            if keyword in question_lower:
                concepts.append(keyword)
        
        # Extract noun phrases (simple heuristic)
        words = question_lower.split()
        for i, word in enumerate(words):
            # Look for "accidental taking", "external storage", etc.
            if i < len(words) - 1:
                phrase = f"{word} {words[i+1]}"
                concepts.append(phrase)
        
        return list(set(concepts))[:10]  # Remove duplicates, limit to 10
    
    def _build_context_from_graph(self, related_info: List[Dict], question: str) -> str:
        """Build textual context from graph relationships."""
        context_parts = []
        
        for info in related_info:
            entity = info["entity"]
            entity_name = entity.get("name", entity.get("id"))
            entity_type = entity.get("type", "Entity")
            
            context_parts.append(f"\n=== {entity_type}: {entity_name} ===\n")
            
            # Add entity description if available
            props = entity.get("properties", {})
            if "description" in props:
                context_parts.append(f"Description: {props['description']}\n")
            
            # Add relationships
            relationships = info["relationships"]
            
            if relationships["outgoing"]:
                context_parts.append("Relationships (requires/contains):")
                for rel in relationships["outgoing"][:5]:
                    rel_type = rel["relationship"]
                    target_name = rel["target_name"]
                    context_parts.append(f"  - {entity_name} {rel_type} {target_name}")
                context_parts.append("")
            
            if relationships["incoming"]:
                context_parts.append("Related from:")
                for rel in relationships["incoming"][:5]:
                    rel_type = rel["relationship"]
                    source_name = rel["source_name"]
                    context_parts.append(f"  - {source_name} {rel_type} {entity_name}")
                context_parts.append("")
            
            # Add traversal results
            traversal = info.get("traversal", {})
            if traversal.get("related"):
                context_parts.append("Related entities (2-hop):")
                for related in traversal["related"][:5]:
                    rel_type = related.get("relationship", "relates_to")
                    related_name = related.get("entity_name", related.get("entity_id"))
                    context_parts.append(f"  - {entity_name} → {related_name} ({rel_type})")
                context_parts.append("")
        
        return "\n".join(context_parts)

