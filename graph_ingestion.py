from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import networkx as nx
import json
import os
import pickle
from typing import List, Dict, Any
from huggingface_hub import InferenceClient
import os as os_module
import requests
from urllib.parse import urlparse

class GraphIngestion:
    def __init__(self, graph_db_path="./graph_db"):
        self.graph_db_path = graph_db_path
        os.makedirs(graph_db_path, exist_ok=True)
        
        # Initialize graph
        self.graph = nx.DiGraph()
        
        # Initialize LLM client for entity extraction
        hf_token = os_module.getenv("HUGGINGFACE_API_KEY")
        if hf_token:
            self.llm_client = InferenceClient(token=hf_token)
            self.model = "openai/gpt-oss-20b"
        else:
            self.llm_client = None
            self.model = None
    
    def load_document(self, file_path: str) -> List[str]:
        """Load document and return text chunks. Supports TXT, PDF files."""
        try:
            # Detect file type
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # Load based on file type
            if file_ext == '.pdf':
                try:
                    loader = PyPDFLoader(file_path)
                except Exception:
                    # Fallback to unstructured if PyPDF fails
                    try:
                        loader = UnstructuredPDFLoader(file_path)
                    except Exception:
                        raise Exception("PDF loading failed. Install: pip install pypdf unstructured")
            else:
                # Default to text loader
                loader = TextLoader(file_path)
            
            documents = loader.load()
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = text_splitter.split_documents(documents)
            
            return [chunk.page_content for chunk in chunks]
        except Exception as e:
            raise Exception(f"Error loading document {file_path}: {str(e)}")
    
    def extract_entities_llm(self, text_chunk: str) -> Dict[str, Any]:
        """Extract entities and relationships from text using LLM."""
        if not self.llm_client:
            return {"entities": [], "relationships": []}
        
        prompt = f"""Extract entities and relationships from this legal/policy text. Focus on extracting:

1. CRIMES/VIOLATIONS (e.g., "Theft", "Fraud", "Assault")
2. LEGAL ELEMENTS (e.g., "Intent", "Act", "Knowledge")
3. LEGAL CONCEPTS (e.g., "Accidental Act", "Mistake of Fact", "Consent")
4. REQUIREMENTS/POLICIES (e.g., "DPO Approval", "Encryption Required")
5. SECTIONS/SUBSECTIONS (e.g., "Section 4.2", "Section 5.1")
6. RELATIONSHIPS (REQUIRES, LACKS, IS_NOT, CONTAINS, RELATES_TO)

Text to analyze:
{text_chunk}

Respond ONLY with a valid JSON object in this format:
{{
  "entities": [
    {{"id": "entity1", "type": "Crime|Element|Concept|Requirement|Section", "name": "entity name", "properties": {{"description": "...", "section": "..."}}}},
    {{"id": "entity2", "type": "...", "name": "...", "properties": {{}}}}
  ],
  "relationships": [
    {{"source": "entity1", "target": "entity2", "type": "REQUIRES|LACKS|IS_NOT|CONTAINS|RELATES_TO", "properties": {{}}}}
  ]
}}

Output ONLY the JSON, no other text:"""
        
        try:
            # Try text generation
            try:
                response = self.llm_client.text_generation(
                    prompt=prompt,
                    model=self.model,
                    max_new_tokens=2000,
                    temperature=0.1,
                    return_full_text=False
                )
                response_text = str(response).strip()
            except Exception:
                # Fallback to chat completion
                messages = [{"role": "user", "content": prompt}]
                response = self.llm_client.chat_completion(
                    messages=messages,
                    model=self.model,
                    max_tokens=2000,
                    temperature=0.1
                )
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    if hasattr(response.choices[0], 'message'):
                        response_text = response.choices[0].message.content
                    else:
                        response_text = str(response.choices[0])
                else:
                    return {"entities": [], "relationships": []}
            
            # Clean response
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:].strip()
            elif response_text.startswith("```"):
                response_text = response_text[3:].strip()
            if response_text.endswith("```"):
                response_text = response_text[:-3].strip()
            
            # Extract JSON
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)
                return result
            else:
                return {"entities": [], "relationships": []}
                
        except Exception as e:
            return {"entities": [], "relationships": []}
    
    def extract_entities_rule_based(self, text_chunk: str, source_file: str) -> Dict[str, Any]:
        """Fallback rule-based extraction for when LLM is not available."""
        entities = []
        relationships = []
        
        lines = text_chunk.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Extract sections (e.g., "Section 4:", "4.1", "Section 4.2")
            import re
            section_match = re.match(r'^[Ss]ection\s+(\d+(?:\.\d+)?)|^(\d+\.\d+)\s*[:\-]', line)
            if section_match:
                section_num = section_match.group(1) or section_match.group(2)
                section_id = f"section_{section_num.replace('.', '_')}"
                current_section = section_id
                
                # Extract section title
                title_match = re.search(r'[:\-]\s*(.+)', line)
                title = title_match.group(1).strip() if title_match else f"Section {section_num}"
                
                entities.append({
                    "id": section_id,
                    "type": "Section",
                    "name": f"Section {section_num}",
                    "properties": {
                        "section_number": section_num,
                        "title": title,
                        "source_file": source_file,
                        "content": line
                    }
                })
            
            # Extract requirements (numbered items like "4.1", "4.2", "5.1")
            req_match = re.match(r'^(\d+\.\d+)\s+(.+)', line)
            if req_match:
                req_num = req_match.group(1)
                req_text = req_match.group(2).strip()
                req_id = f"req_{req_num.replace('.', '_')}"
                
                entities.append({
                    "id": req_id,
                    "type": "Requirement",
                    "name": f"Requirement {req_num}",
                    "properties": {
                        "requirement_number": req_num,
                        "text": req_text,
                        "source_file": source_file
                    }
                })
                
                # Link requirement to section
                if current_section:
                    relationships.append({
                        "source": current_section,
                        "target": req_id,
                        "type": "CONTAINS",
                        "properties": {}
                    })
                
                # Extract keywords for relationships
                if "require" in req_text.lower() or "must" in req_text.lower():
                    # Find what it requires
                    if "approval" in req_text.lower() and "dpo" in req_text.lower():
                        entities.append({
                            "id": "req_dpo_approval",
                            "type": "Requirement",
                            "name": "DPO Approval Required",
                            "properties": {"source_file": source_file}
                        })
                        relationships.append({
                            "source": req_id,
                            "target": "req_dpo_approval",
                            "type": "REQUIRES",
                            "properties": {}
                        })
        
        return {"entities": entities, "relationships": relationships}
    
    def add_to_graph(self, entities: List[Dict], relationships: List[Dict], source_file: str):
        """Add extracted entities and relationships to the graph."""
        # Add entities as nodes
        for entity in entities:
            node_id = entity.get("id")
            if not node_id:
                continue
            
            # Merge properties with existing node if it exists
            if self.graph.has_node(node_id):
                existing_props = self.graph.nodes[node_id]
                # Update properties, keeping existing ones
                for key, value in entity.get("properties", {}).items():
                    if key not in existing_props or not existing_props[key]:
                        existing_props[key] = value
                existing_props["source_files"] = existing_props.get("source_files", set())
                if isinstance(existing_props["source_files"], str):
                    existing_props["source_files"] = {existing_props["source_files"]}
                existing_props["source_files"].add(source_file)
            else:
                # Create new node
                props = entity.get("properties", {}).copy()
                props["source_files"] = {source_file}
                props["entity_type"] = entity.get("type", "Unknown")
                props["entity_name"] = entity.get("name", node_id)
                self.graph.add_node(node_id, **props)
        
        # Add relationships as edges
        for rel in relationships:
            source = rel.get("source")
            target = rel.get("target")
            rel_type = rel.get("type", "RELATES_TO")
            
            if source and target:
                # Merge properties
                if self.graph.has_edge(source, target):
                    existing_props = self.graph.edges[source, target]
                    existing_props["relationship_types"] = existing_props.get("relationship_types", [])
                    if isinstance(existing_props["relationship_types"], str):
                        existing_props["relationship_types"] = [existing_props["relationship_types"]]
                    if rel_type not in existing_props["relationship_types"]:
                        existing_props["relationship_types"].append(rel_type)
                else:
                    props = rel.get("properties", {}).copy()
                    props["relationship_type"] = rel_type
                    props["source_files"] = {source_file}
                    self.graph.add_edge(source, target, **props)
    
    def ingest_file(self, file_path: str, use_llm: bool = True) -> Dict[str, Any]:
        """Ingest a single file and extract graph structure."""
        source_file = os.path.basename(file_path)
        chunks = self.load_document(file_path)
        
        total_entities = 0
        total_relationships = 0
        
        for chunk in chunks:
            if use_llm and self.llm_client:
                extraction_result = self.extract_entities_llm(chunk)
            else:
                extraction_result = self.extract_entities_rule_based(chunk, source_file)
            
            entities = extraction_result.get("entities", [])
            relationships = extraction_result.get("relationships", [])
            
            if entities or relationships:
                self.add_to_graph(entities, relationships, source_file)
                total_entities += len(entities)
                total_relationships += len(relationships)
        
        return {
            "file": source_file,
            "chunks_processed": len(chunks),
            "entities_extracted": total_entities,
            "relationships_extracted": total_relationships,
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges()
        }
    
    def ingest_folder(self, folder_path: str, use_llm: bool = True) -> Dict[str, Any]:
        """Ingest all documents from a folder."""
        results = []
        
        if not os.path.exists(folder_path):
            raise Exception(f"Folder {folder_path} does not exist")
        
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            if os.path.isfile(file_path) and filename.endswith('.txt'):
                try:
                    result = self.ingest_file(file_path, use_llm=use_llm)
                    results.append(result)
                except Exception as e:
                    results.append({
                        "file": filename,
                        "error": str(e)
                    })
        
        # Save graph
        self.save_graph()
        
        return {
            "files_processed": len(results),
            "results": results,
            "graph_stats": {
                "total_nodes": self.graph.number_of_nodes(),
                "total_edges": self.graph.number_of_edges()
            }
        }
    
    def ingest_from_url(self, url: str, use_llm: bool = True) -> Dict[str, Any]:
        """Download document from URL and ingest into graph. Supports PDF, TXT, HTML, etc."""
        try:
            # Download from URL
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, timeout=60, allow_redirects=True, headers=headers)
            response.raise_for_status()
            
            # Get filename from URL or use default
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path) or "downloaded_document"
            
            # Detect file type from URL or content
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' in content_type or filename.endswith('.pdf'):
                file_ext = '.pdf'
            elif filename.endswith('.txt'):
                file_ext = '.txt'
            elif filename.endswith('.html') or filename.endswith('.htm'):
                file_ext = '.html'
            else:
                # Default to pdf for the IPC link, or try to detect
                if 'pdf' in url.lower() or url.endswith('.pdf'):
                    file_ext = '.pdf'
                else:
                    file_ext = '.txt'
            
            if not filename.endswith(file_ext):
                filename = filename.split('.')[0] + file_ext
            
            # Save to temp directory
            temp_dir = os.path.join(self.graph_db_path, "temp")
            os.makedirs(temp_dir, exist_ok=True)
            
            temp_file = os.path.join(temp_dir, filename)
            
            # Save file based on type
            if file_ext == '.pdf':
                # Save PDF as binary
                with open(temp_file, "wb") as f:
                    f.write(response.content)
            else:
                # Save as text
                try:
                    content = response.text
                    with open(temp_file, "w", encoding="utf-8") as f:
                        f.write(content)
                except UnicodeDecodeError:
                    # If can't decode as text, save as binary anyway
                    with open(temp_file, "wb") as f:
                        f.write(response.content)
            
            try:
                # Try to ingest the file
                result = self.ingest_file(temp_file, use_llm=use_llm)
                result["source_url"] = url
                result["file_type"] = file_ext
                
                # Clean up temp file
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                
                return result
            except Exception as e:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                raise Exception(f"Error ingesting downloaded file: {str(e)}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error downloading from URL: {str(e)}")
    
    def ingest_text(self, text: str, source_name: str = "uploaded_text", use_llm: bool = True) -> Dict[str, Any]:
        """Ingest text directly (for manual uploads)."""
        # Save text to temporary file
        temp_dir = os.path.join(self.graph_db_path, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_file = os.path.join(temp_dir, f"{source_name}.txt")
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(text)
        
        try:
            result = self.ingest_file(temp_file, use_llm=use_llm)
            # Clean up temp file
            os.remove(temp_file)
            return result
        except Exception as e:
            if os.path.exists(temp_file):
                os.remove(temp_file)
            raise e
    
    def save_graph(self):
        """Save graph to disk."""
        graph_file = os.path.join(self.graph_db_path, "knowledge_graph.pkl")
        
        # Convert sets to lists for JSON serialization
        graph_copy = self.graph.copy()
        for node in graph_copy.nodes():
            props = graph_copy.nodes[node]
            if "source_files" in props and isinstance(props["source_files"], set):
                props["source_files"] = list(props["source_files"])
        
        for source, target in graph_copy.edges():
            props = graph_copy.edges[source, target]
            if "source_files" in props and isinstance(props["source_files"], set):
                props["source_files"] = list(props["source_files"])
            if "relationship_types" in props and isinstance(props["relationship_types"], set):
                props["relationship_types"] = list(props["relationship_types"])
        
        with open(graph_file, "wb") as f:
            pickle.dump(graph_copy, f)
        
        # Also save as JSON for inspection
        json_file = os.path.join(self.graph_db_path, "knowledge_graph.json")
        graph_data = {
            "nodes": [
                {
                    "id": node,
                    **{k: (list(v) if isinstance(v, set) else v) for k, v in props.items()}
                }
                for node, props in graph_copy.nodes(data=True)
            ],
            "edges": [
                {
                    "source": source,
                    "target": target,
                    **{k: (list(v) if isinstance(v, set) else v) for k, v in props.items()}
                }
                for source, target, props in graph_copy.edges(data=True)
            ]
        }
        
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
    
    def load_graph(self):
        """Load graph from disk."""
        graph_file = os.path.join(self.graph_db_path, "knowledge_graph.pkl")
        
        if os.path.exists(graph_file):
            with open(graph_file, "rb") as f:
                self.graph = pickle.load(f)
                # Convert lists back to sets
                for node in self.graph.nodes():
                    props = self.graph.nodes[node]
                    if "source_files" in props and isinstance(props["source_files"], list):
                        props["source_files"] = set(props["source_files"])
                
                for source, target in self.graph.edges():
                    props = self.graph.edges[source, target]
                    if "source_files" in props and isinstance(props["source_files"], list):
                        props["source_files"] = set(props["source_files"])
            return True
        return False
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the graph."""
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_types": self._count_node_types(),
            "relationship_types": self._count_relationship_types()
        }
    
    def _count_node_types(self) -> Dict[str, int]:
        """Count nodes by type."""
        types = {}
        for node, props in self.graph.nodes(data=True):
            node_type = props.get("entity_type", "Unknown")
            types[node_type] = types.get(node_type, 0) + 1
        return types
    
    def _count_relationship_types(self) -> Dict[str, int]:
        """Count edges by relationship type."""
        types = {}
        for source, target, props in self.graph.edges(data=True):
            rel_type = props.get("relationship_type", "RELATES_TO")
            types[rel_type] = types.get(rel_type, 0) + 1
        return types

