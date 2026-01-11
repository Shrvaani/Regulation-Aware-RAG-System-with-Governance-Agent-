import streamlit as st
import json
import os
import tempfile
from pathlib import Path

# Try to load environment variables from .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
except Exception:
    pass

# Import all necessary modules
from graph_ingestion import GraphIngestion
from graph_query_system import GraphQuerySystem
from regulation_aware_rag import RegulationAwareRAG
from document_ingestion import DocumentIngestion
from sample_policies import create_sample_policies

st.set_page_config(
    page_title="RAG Governance System",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ RAG Governance System")

# Sidebar for information
with st.sidebar:
    st.header("About")
    st.markdown("""
    **RAG Governance System** provides:
    1. **Graph Ingestion**: Build knowledge graphs from documents
    2. **Graph Query**: Query the knowledge graph for answers
    3. **RAG Evaluation**: Evaluate actions using vector RAG
    """)

# Initialize systems using session state for caching
@st.cache_resource
def initialize_rag_system():
    """Initialize the RAG system for vector-based evaluation."""
    # Check if vector database exists, if not create sample policies and ingest
    if not os.path.exists("./chroma_db"):
        if not os.path.exists("./policy_documents"):
            create_sample_policies()
        else:
            txt_files = [f for f in os.listdir("./policy_documents") if f.endswith('.txt')]
            if not txt_files:
                create_sample_policies()
        
        # Ingest policies into vector database
        ingestion = DocumentIngestion()
        try:
            ingestion.ingest_folder("./policy_documents")
        except Exception:
            pass
    
    try:
        return RegulationAwareRAG()
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {str(e)}")
        return None

@st.cache_resource
def initialize_graph_ingestion():
    """Initialize the graph ingestion system."""
    try:
        graph_ingestion = GraphIngestion()
        # Try to load existing graph if available
        graph_ingestion.load_graph()
        return graph_ingestion
    except Exception as e:
        st.error(f"Failed to initialize graph ingestion: {str(e)}")
        return None

def get_graph_query_system():
    """Get graph query system (reinitialize to get latest graph state)."""
    try:
        # Reinitialize to get latest graph state
        graph_ingestion = initialize_graph_ingestion()
        if graph_ingestion and graph_ingestion.graph.number_of_nodes() > 0:
            return GraphQuerySystem()
        return None
    except Exception:
        return None

# Initialize systems
rag_system = initialize_rag_system()
graph_ingestion = initialize_graph_ingestion()

# Tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["📊 Graph Ingestion", "🔍 Graph Query", "⚖️ RAG Evaluation"])

# Tab 1: Graph Ingestion
with tab1:
    st.header("📊 Graph RAG Data Ingestion")
    
    if graph_ingestion is None:
        st.error("⚠️ Graph ingestion system not initialized. Please check your configuration.")
    else:
        ingestion_method = st.radio(
            "Select Ingestion Method",
            ["URL", "File Upload", "Text Input", "Folder Path"],
            horizontal=True
        )
        
        use_llm = st.checkbox("Use LLM for entity extraction (slower but more accurate)", value=False)
        
        if ingestion_method == "URL":
            with st.form("ingest_url_form"):
                st.subheader("Ingest from URL")
                url = st.text_input(
                    "Document URL",
                    placeholder="https://www.example.com/document.pdf",
                    help="Enter URL of the document to ingest"
                )
                
                submitted_url = st.form_submit_button("Ingest URL", type="primary", use_container_width=True)
                
                if submitted_url:
                    if not url:
                        st.error("Please enter a URL.")
                    else:
                        with st.spinner(f"Ingesting document from URL... This may take a while."):
                            try:
                                result = graph_ingestion.ingest_from_url(url, use_llm=use_llm)
                                graph_ingestion.save_graph()
                                
                                st.success("✅ Document ingested successfully!")
                                st.json(result)
                                
                                # Clear cache to reload graph
                                st.cache_resource.clear()
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")
        
        elif ingestion_method == "File Upload":
            with st.form("ingest_file_form"):
                st.subheader("Upload File")
                uploaded_file = st.file_uploader(
                    "Choose a file (PDF or TXT)",
                    type=['pdf', 'txt'],
                    help="Upload a PDF or TXT file to ingest"
                )
                
                submitted_file = st.form_submit_button("Upload & Ingest", type="primary", use_container_width=True)
                
                if submitted_file:
                    if uploaded_file is None:
                        st.error("Please upload a file.")
                    else:
                        with st.spinner("Uploading and ingesting file... This may take a while."):
                            try:
                                # Save uploaded file to temporary location
                                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                                    tmp_file.write(uploaded_file.getvalue())
                                    tmp_path = tmp_file.name
                                
                                try:
                                    result = graph_ingestion.ingest_file(tmp_path, use_llm=use_llm)
                                    graph_ingestion.save_graph()
                                    
                                    st.success("✅ File ingested successfully!")
                                    st.json(result)
                                    
                                    # Clear cache to reload graph
                                    st.cache_resource.clear()
                                finally:
                                    # Clean up temporary file
                                    if os.path.exists(tmp_path):
                                        os.remove(tmp_path)
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")
        
        elif ingestion_method == "Text Input":
            with st.form("ingest_text_form"):
                st.subheader("Enter Text")
                source_name = st.text_input(
                    "Source Name",
                    value="uploaded_text",
                    help="Name for this text source"
                )
                text_input = st.text_area(
                    "Text to Ingest",
                    placeholder="Enter text content here...",
                    height=200,
                    help="Enter the text content to ingest into the graph"
                )
                
                submitted_text = st.form_submit_button("Ingest Text", type="primary", use_container_width=True)
                
                if submitted_text:
                    if not text_input:
                        st.error("Please enter some text.")
                    else:
                        with st.spinner("Ingesting text... This may take a while."):
                            try:
                                result = graph_ingestion.ingest_text(text_input, source_name=source_name, use_llm=use_llm)
                                graph_ingestion.save_graph()
                                
                                st.success("✅ Text ingested successfully!")
                                st.json(result)
                                
                                # Clear cache to reload graph
                                st.cache_resource.clear()
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")
        
        else:  # Folder Path
            with st.form("ingest_folder_form"):
                st.subheader("Ingest Folder")
                folder_path = st.text_input(
                    "Folder Path",
                    value="./policy_documents",
                    help="Enter path to folder containing documents"
                )
                
                submitted_folder = st.form_submit_button("Ingest Folder", type="primary", use_container_width=True)
                
                if submitted_folder:
                    if not folder_path:
                        st.error("Please enter a folder path.")
                    else:
                        with st.spinner(f"Ingesting folder '{folder_path}'... This may take a while."):
                            try:
                                result = graph_ingestion.ingest_folder(folder_path, use_llm=use_llm)
                                
                                st.success("✅ Folder ingested successfully!")
                                st.json(result)
                                
                                # Clear cache to reload graph
                                st.cache_resource.clear()
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")

# Tab 2: Graph Query
with tab2:
    st.header("🔍 Query Knowledge Graph")
    
    with st.form("query_graph_form"):
        question = st.text_input(
            "Question *",
            placeholder="e.g., I accidentally took my colleague's mobile, but it was not intentional. Is this a crime?",
            help="Enter your question about the ingested documents"
        )
        
        context = st.text_area(
            "Additional Context (Optional)",
            placeholder="Provide any additional context for the question...",
            height=100,
            help="Optional context that might help answer the question"
        )
        
        submitted_query = st.form_submit_button("Query Graph", type="primary", use_container_width=True)
    
    if submitted_query:
        if not question:
            st.error("Please enter a question.")
        else:
            with st.spinner("Querying knowledge graph... Please wait."):
                try:
                    query_system = get_graph_query_system()
                    
                    if query_system is None:
                        st.warning("⚠️ Knowledge graph is empty. Please ingest documents first using the 'Graph Ingestion' tab.")
                        st.info("Go to the 'Graph Ingestion' tab to ingest documents.")
                    else:
                        result = query_system.answer_question(question, context if context else "")
                        
                        st.success("Query complete!")
                        
                        # Answer Section
                        st.markdown("### Answer")
                        st.info(result.get('answer', 'N/A'))
                        
                        # Decision Section
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            decision = result.get('decision', 'Unknown')
                            decision_emoji = {
                                "Is a Crime": "🔴",
                                "Not a Crime": "🟢",
                                "Conditional": "🟡",
                                "Unknown": "⚪"
                            }.get(decision, "⚪")
                            st.markdown(f"#### {decision_emoji} Decision: **{decision}**")
                        
                        with col2:
                            risk_level = result.get('risk_level', 'N/A')
                            risk_emoji = {
                                "Low": "🟢",
                                "Medium": "🟡",
                                "High": "🟠",
                                "Critical": "🔴"
                            }.get(risk_level, "⚪")
                            st.markdown(f"#### {risk_emoji} **{risk_level} Risk**")
                        
                        # Confidence Score
                        confidence = result.get('confidence_score', 0.0) * 100
                        st.progress(confidence / 100, text=f"Confidence Score: {confidence:.1f}%")
                        
                        st.divider()
                        
                        # Matched Entities
                        matched_entities = result.get('matched_entities', [])
                        if matched_entities:
                            st.markdown("#### Matched Entities")
                            for entity in matched_entities[:5]:
                                entity_name = entity.get('name', entity.get('id', 'Unknown'))
                                entity_type = entity.get('type', 'Unknown')
                                st.markdown(f"- **{entity_type}**: {entity_name}")
                        
                        # Graph Paths
                        graph_paths = result.get('graph_paths', [])
                        if graph_paths:
                            st.markdown("#### Relationship Paths")
                            for path in graph_paths[:5]:
                                source = path.get('source', 'Unknown')
                                relationship = path.get('relationship', 'RELATES_TO')
                                target = path.get('target', 'Unknown')
                                st.markdown(f"- **{source}** → `{relationship}` → **{target}**")
                        
                        # References
                        references = result.get('references', [])
                        if references:
                            st.markdown("#### References")
                            for ref in references:
                                st.markdown(f"- {ref}")
                        
                        # Expandable JSON view
                        with st.expander("View Raw JSON Response"):
                            st.json(result)
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# Tab 3: RAG Evaluation (Original functionality)
with tab3:
    st.header("⚖️ RAG Evaluation")
    
    if rag_system is None:
        st.error("⚠️ RAG system not initialized. Please check your configuration.")
    else:
        with st.form("evaluate_form"):
            action = st.text_input(
                "Action to Evaluate *",
                placeholder="e.g., Store user data on an external analytics server",
                help="Enter the action you want to evaluate for compliance"
            )
            
            context = st.text_area(
                "Additional Context (Optional)",
                placeholder="Provide any additional context for the evaluation...",
                height=100,
                help="Optional context that might help with the evaluation"
            )
            
            submitted = st.form_submit_button("Evaluate Action", type="primary", use_container_width=True)

        if submitted:
            if not action:
                st.error("Please enter an action to evaluate.")
            else:
                with st.spinner("Evaluating action... Please wait."):
                    try:
                        result = rag_system.process_action(action, context if context else "")
                        
                        st.success("Evaluation complete!")
                        
                        # Decision Section
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            decision_color = {
                                "Allowed": "🟢",
                                "Not Allowed": "🔴",
                                "Conditional": "🟡"
                            }
                            decision_emoji = decision_color.get(result.get("decision", ""), "⚪")
                            st.markdown(f"### {decision_emoji} Decision: **{result.get('decision', 'N/A')}**")
                        
                        with col2:
                            risk_color = {
                                "Low": "🟢",
                                "Medium": "🟡",
                                "High": "🟠",
                                "Critical": "🔴"
                            }
                            risk_level = result.get('risk_level', 'N/A')
                            risk_emoji = risk_color.get(risk_level, "⚪")
                            st.markdown(f"### {risk_emoji} **{risk_level} Risk**")
                        
                        # Confidence Score
                        confidence = result.get('confidence_score', 0.0) * 100
                        st.progress(confidence / 100, text=f"Confidence Score: {confidence:.1f}%")
                        
                        st.divider()
                        
                        # Action
                        st.markdown("#### Action")
                        st.info(result.get('action', 'N/A'))
                        
                        # Reason
                        st.markdown("#### Reason")
                        st.write(result.get('reason', 'N/A'))
                        
                        # Suggested Changes
                        suggested_changes = result.get('suggested_changes', [])
                        if suggested_changes:
                            st.markdown("#### Suggested Changes")
                            for i, change in enumerate(suggested_changes, 1):
                                st.markdown(f"{i}. {change}")
                        
                        # Alternative Actions
                        alternative_actions = result.get('alternative_actions', [])
                        if alternative_actions:
                            st.markdown("#### Alternative Actions")
                            for i, alt_action in enumerate(alternative_actions, 1):
                                st.markdown(f"{i}. {alt_action}")
                        
                        # Policy References
                        references = result.get('references', [])
                        if references:
                            st.markdown("#### Policy References")
                            for ref in references:
                                st.markdown(f"- {ref}")
                        
                        # Policy Sources
                        st.markdown("#### Policy Sources")
                        col3, col4 = st.columns([2, 1])
                        with col3:
                            policy_sources = result.get('policy_sources', [])
                            if policy_sources:
                                st.markdown(", ".join(policy_sources))
                            else:
                                st.markdown("N/A")
                        with col4:
                            retrieved_count = result.get('retrieved_policy_count', 0)
                            st.caption(f"Retrieved {retrieved_count} policy sections")
                        
                        # Expandable JSON view
                        with st.expander("View Raw JSON Response"):
                            st.json(result)
                            
                    except Exception as e:
                        st.error(f"❌ An error occurred: {str(e)}")
