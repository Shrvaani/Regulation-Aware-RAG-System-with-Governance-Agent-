import streamlit as st
import requests
import json
import os

st.set_page_config(
    page_title="RAG Governance System",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ RAG Governance System")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    api_url = st.text_input(
        "API URL",
        value="http://localhost:5000",
        help="URL of the Flask API server"
    )
    st.markdown("---")
    st.markdown("### How to use")
    st.markdown("""
    1. Make sure the Flask API server is running:
       ```bash
       python api_server.py
       ```
    2. **Graph Ingestion**: Ingest documents to build knowledge graph
    3. **Graph Query**: Ask questions using the knowledge graph
    4. **RAG Evaluation**: Evaluate actions using vector RAG
    """)

# Tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["📊 Graph Ingestion", "🔍 Graph Query", "⚖️ RAG Evaluation"])

# Tab 1: Graph Ingestion
with tab1:
    st.header("📊 Graph RAG Data Ingestion")
    
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
                            response = requests.post(
                                f"{api_url}/graph/ingest-url",
                                json={"url": url, "use_llm": use_llm},
                                timeout=300
                            )
                            
                            if response.status_code == 200:
                                result = response.json()
                                st.success("✅ Document ingested successfully!")
                                st.json(result)
                            else:
                                error_data = response.json() if response.content else {"error": "Unknown error"}
                                st.error(f"Error: {error_data.get('error', 'Failed to ingest')}")
                        except requests.exceptions.ConnectionError:
                            st.error(f"❌ Cannot connect to API at {api_url}")
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
                            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                            data = {"use_llm": use_llm}
                            response = requests.post(
                                f"{api_url}/graph/ingest-file",
                                files=files,
                                data=data,
                                timeout=300
                            )
                            
                            if response.status_code == 200:
                                result = response.json()
                                st.success("✅ File ingested successfully!")
                                st.json(result)
                            else:
                                error_data = response.json() if response.content else {"error": "Unknown error"}
                                st.error(f"Error: {error_data.get('error', 'Failed to ingest')}")
                        except requests.exceptions.ConnectionError:
                            st.error(f"❌ Cannot connect to API at {api_url}")
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
                            response = requests.post(
                                f"{api_url}/graph/ingest-text",
                                json={"text": text_input, "source_name": source_name, "use_llm": use_llm},
                                timeout=300
                            )
                            
                            if response.status_code == 200:
                                result = response.json()
                                st.success("✅ Text ingested successfully!")
                                st.json(result)
                            else:
                                error_data = response.json() if response.content else {"error": "Unknown error"}
                                st.error(f"Error: {error_data.get('error', 'Failed to ingest')}")
                        except requests.exceptions.ConnectionError:
                            st.error(f"❌ Cannot connect to API at {api_url}")
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
                            response = requests.post(
                                f"{api_url}/graph/ingest-folder",
                                json={"folder_path": folder_path, "use_llm": use_llm},
                                timeout=300
                            )
                            
                            if response.status_code == 200:
                                result = response.json()
                                st.success("✅ Folder ingested successfully!")
                                st.json(result)
                            else:
                                error_data = response.json() if response.content else {"error": "Unknown error"}
                                st.error(f"Error: {error_data.get('error', 'Failed to ingest')}")
                        except requests.exceptions.ConnectionError:
                            st.error(f"❌ Cannot connect to API at {api_url}")
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
                    response = requests.post(
                        f"{api_url}/graph/query",
                        json={"question": question, "context": context if context else ""},
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
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
                    
                    elif response.status_code == 400:
                        error_data = response.json() if response.content else {"error": "Unknown error"}
                        st.warning(f"⚠️ {error_data.get('error', 'Graph is empty. Please ingest documents first.')}")
                        st.info("Go to the 'Graph Ingestion' tab to ingest documents.")
                    else:
                        error_data = response.json() if response.content else {"error": "Unknown error"}
                        st.error(f"Error: {error_data.get('error', 'Failed to query graph')}")
                
                except requests.exceptions.ConnectionError:
                    st.error(f"❌ Cannot connect to API at {api_url}")
                except requests.exceptions.Timeout:
                    st.error("⏱️ Request timed out. Please try again.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# Tab 3: RAG Evaluation (Original functionality)
with tab3:
    st.header("⚖️ RAG Evaluation")
    
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
                    response = requests.post(
                        f"{api_url}/evaluate",
                        json={
                            "action": action,
                            "context": context if context else ""
                        },
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
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
                            
                    else:
                        error_data = response.json() if response.content else {"error": "Unknown error"}
                        st.error(f"Error: {error_data.get('error', 'Failed to evaluate action')}")
                        st.json(error_data)
                        
                except requests.exceptions.ConnectionError:
                    st.error(f"❌ Cannot connect to API at {api_url}. Please make sure the Flask server is running.")
                    st.info("Start the server with: `python api_server.py`")
                except requests.exceptions.Timeout:
                    st.error("⏱️ Request timed out. The evaluation is taking too long. Please try again.")
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")
# Health check section at the bottom
with st.expander("🔍 API Health Check"):
    if st.button("Check API Health"):
        try:
            health_response = requests.get(f"{api_url}/health", timeout=5)
            if health_response.status_code == 200:
                health_data = health_response.json()
                if health_data.get('system_ready'):
                    st.success("✅ API is healthy and ready")
                else:
                    st.warning("⚠️ API is running but system is not ready")
                st.json(health_data)
            else:
                st.error(f"API returned status code: {health_response.status_code}")
        except requests.exceptions.ConnectionError:
            st.error(f"Cannot connect to API at {api_url}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
