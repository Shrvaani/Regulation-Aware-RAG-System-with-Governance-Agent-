from flask import Flask, request, jsonify, render_template
from regulation_aware_rag import RegulationAwareRAG
from document_ingestion import DocumentIngestion
from graph_ingestion import GraphIngestion
from graph_query_system import GraphQuerySystem
import os

# Try to load environment variables from .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
except Exception:
    pass

app = Flask(__name__)
rag_system = None
graph_ingestion = None
graph_query_system = None

def initialize_system():
    global rag_system
    
    if not os.path.exists("./chroma_db"):
        print("Creating vector database...")
        
        # Check if policy_documents folder exists and has files
        if not os.path.exists("./policy_documents"):
            from sample_policies import create_sample_policies
            create_sample_policies()
        else:
            txt_files = [f for f in os.listdir("./policy_documents") if f.endswith('.txt')]
            if not txt_files:
                from sample_policies import create_sample_policies
                create_sample_policies()
        
        # Ingest policies into vector database
        ingestion = DocumentIngestion()
        try:
            ingestion.ingest_folder("./policy_documents")
        except Exception:
            pass
    
    rag_system = RegulationAwareRAG()
    
    # Initialize Graph RAG ingestion (optional - for testing)
    global graph_ingestion, graph_query_system
    try:
        graph_ingestion = GraphIngestion()
        # Try to load existing graph if available
        graph_ingestion.load_graph()
        
        # Initialize Graph Query System only if graph exists or can be created
        if graph_ingestion.graph.number_of_nodes() > 0:
            try:
                graph_query_system = GraphQuerySystem()
            except Exception:
                graph_query_system = None
        else:
            graph_query_system = None
    except Exception:
        graph_ingestion = None
        graph_query_system = None

@app.route('/evaluate', methods=['POST'])
def evaluate_action():
    try:
        data = request.get_json()
        
        if not data or 'action' not in data:
            return jsonify({"error": "Missing required field: 'action'"}), 400
        
        result = rag_system.process_action(
            action=data['action'],
            context=data.get('context', '')
        )
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/batch-evaluate', methods=['POST'])
def batch_evaluate():
    try:
        data = request.get_json()
        
        if not data or 'actions' not in data:
            return jsonify({"error": "Missing 'actions' field"}), 400
        
        results = []
        for action_data in data['actions']:
            result = rag_system.process_action(
                action=action_data.get('action'),
                context=action_data.get('context', '')
            )
            results.append(result)
        
        return jsonify({"results": results, "total_evaluated": len(results)}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "system_ready": rag_system is not None}), 200

@app.route('/', methods=['GET'])
def index():
    """Simple UI for testing the governance system"""
    return render_template('index.html')

# Graph RAG Ingestion Endpoints

@app.route('/graph/ingest-folder', methods=['POST'])
def ingest_folder_graph():
    """Ingest documents from a folder into graph."""
    try:
        data = request.get_json() or {}
        folder_path = data.get('folder_path', './policy_documents')
        use_llm = data.get('use_llm', True)
        
        if graph_ingestion is None:
            return jsonify({"error": "Graph ingestion not initialized"}), 500
        
        result = graph_ingestion.ingest_folder(folder_path, use_llm=use_llm)
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/graph/ingest-text', methods=['POST'])
def ingest_text_graph():
    """Ingest text directly into graph (manual upload)."""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "Missing required field: 'text'"}), 400
        
        text = data['text']
        source_name = data.get('source_name', 'uploaded_text')
        use_llm = data.get('use_llm', True)
        
        if graph_ingestion is None:
            return jsonify({"error": "Graph ingestion not initialized"}), 500
        
        result = graph_ingestion.ingest_text(text, source_name=source_name, use_llm=use_llm)
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/graph/ingest-file', methods=['POST'])
def ingest_file_graph():
    """Ingest a single file into graph."""
    try:
        data = request.get_json()
        
        if not data or 'file_path' not in data:
            return jsonify({"error": "Missing required field: 'file_path'"}), 400
        
        file_path = data['file_path']
        use_llm = data.get('use_llm', True)
        
        if graph_ingestion is None:
            return jsonify({"error": "Graph ingestion not initialized"}), 500
        
        if not os.path.exists(file_path):
            return jsonify({"error": f"File not found: {file_path}"}), 404
        
        result = graph_ingestion.ingest_file(file_path, use_llm=use_llm)
        graph_ingestion.save_graph()
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/graph/stats', methods=['GET'])
def get_graph_stats():
    """Get graph statistics."""
    try:
        if graph_ingestion is None:
            return jsonify({"error": "Graph ingestion not initialized"}), 500
        
        stats = graph_ingestion.get_graph_stats()
        return jsonify(stats), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/graph/ingest-url', methods=['POST'])
def ingest_url_graph():
    """Ingest document from URL into graph."""
    try:
        data = request.get_json()
        
        if not data or 'url' not in data:
            return jsonify({"error": "Missing required field: 'url'"}), 400
        
        url = data['url']
        use_llm = data.get('use_llm', True)
        
        if graph_ingestion is None:
            return jsonify({"error": "Graph ingestion not initialized"}), 500
        
        result = graph_ingestion.ingest_from_url(url, use_llm=use_llm)
        graph_ingestion.save_graph()
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/graph/view', methods=['GET'])
def view_graph():
    """View graph structure (for testing)."""
    try:
        if graph_ingestion is None:
            return jsonify({"error": "Graph ingestion not initialized"}), 500
        
        graph = graph_ingestion.graph
        
        # Convert to JSON-serializable format
        nodes = [
            {
                "id": node,
                "type": props.get("entity_type", "Unknown"),
                "name": props.get("entity_name", node),
                "properties": {k: (list(v) if isinstance(v, set) else v) 
                             for k, v in props.items() if k != "entity_type" and k != "entity_name"}
            }
            for node, props in graph.nodes(data=True)
        ]
        
        edges = [
            {
                "source": source,
                "target": target,
                "type": props.get("relationship_type", "RELATES_TO"),
                "properties": {k: (list(v) if isinstance(v, set) else v) 
                             for k, v in props.items() if k != "relationship_type"}
            }
            for source, target, props in graph.edges(data=True)
        ]
        
        return jsonify({
            "nodes": nodes[:100],  # Limit to first 100 for preview
            "edges": edges[:100],
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "message": "Showing first 100 nodes and edges. Use /graph/stats for full statistics."
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/graph/query', methods=['POST'])
def query_graph():
    """Query the graph to answer questions."""
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({"error": "Missing required field: 'question'"}), 400
        
        question = data['question']
        context = data.get('context', '')
        
        # Initialize query system lazily (loads latest graph state)
        try:
            query_system = GraphQuerySystem()
            
            # Check if graph has any nodes
            if query_system.graph_retrieval.graph.number_of_nodes() == 0:
                return jsonify({
                    "error": "Knowledge graph is empty. Please ingest documents first using /graph/ingest-url or /graph/ingest-folder"
                }), 400
            
            result = query_system.answer_question(question, context)
            return jsonify(result), 200
            
        except Exception as e:
            return jsonify({"error": f"Error querying graph: {str(e)}"}), 500
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    initialize_system()
    app.run(debug=True, port=5000)