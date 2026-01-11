#!/usr/bin/env python3
"""
Quick script to test Graph RAG ingestion
Run this after starting api_server.py
"""

import requests
import json

BASE_URL = "http://localhost:5000"

def test_ingestion():
    print("=" * 60)
    print("Testing Graph RAG Ingestion")
    print("=" * 60)
    
    # Test 1: Ingest existing policy folder
    print("\n1. Ingesting policy_documents folder...")
    try:
        response = requests.post(
            f"{BASE_URL}/graph/ingest-folder",
            json={
                "folder_path": "./policy_documents",
                "use_llm": False  # Use rule-based extraction (no API key needed)
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print(f"   Files processed: {result.get('files_processed', 0)}")
            print(f"   Total nodes: {result.get('graph_stats', {}).get('total_nodes', 0)}")
            print(f"   Total edges: {result.get('graph_stats', {}).get('total_edges', 0)}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.json())
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure api_server.py is running!")
        return
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # Test 2: Check graph statistics
    print("\n2. Checking graph statistics...")
    try:
        response = requests.get(f"{BASE_URL}/graph/stats")
        if response.status_code == 200:
            stats = response.json()
            print("✅ Graph Statistics:")
            print(f"   Total Nodes: {stats.get('total_nodes', 0)}")
            print(f"   Total Edges: {stats.get('total_edges', 0)}")
            print(f"   Node Types: {stats.get('node_types', {})}")
            print(f"   Relationship Types: {stats.get('relationship_types', {})}")
        else:
            print(f"❌ Error: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: Manual text upload (example)
    print("\n3. Testing manual text upload with legal example...")
    legal_text = """
    Theft requires dishonest intention (mens rea). 
    Essential elements of theft:
    1. Dishonest intention (mens rea) - REQUIRED
    2. Taking movable property
    3. Property of another person
    
    Accidental taking LACKS dishonest intention and therefore IS NOT theft.
    """
    
    try:
        response = requests.post(
            f"{BASE_URL}/graph/ingest-text",
            json={
                "text": legal_text,
                "source_name": "test_theft_concept",
                "use_llm": False
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print(f"   Entities extracted: {result.get('entities_extracted', 0)}")
            print(f"   Relationships extracted: {result.get('relationships_extracted', 0)}")
            print(f"   Total nodes in graph: {result.get('total_nodes', 0)}")
            print(f"   Total edges in graph: {result.get('total_edges', 0)}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.json())
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 4: View graph structure
    print("\n4. Viewing graph structure (first 100 items)...")
    try:
        response = requests.get(f"{BASE_URL}/graph/view")
        if response.status_code == 200:
            graph_data = response.json()
            print(f"✅ Showing {len(graph_data.get('nodes', []))} nodes and {len(graph_data.get('edges', []))} edges")
            print(f"   Total: {graph_data.get('total_nodes', 0)} nodes, {graph_data.get('total_edges', 0)} edges")
            
            # Show sample nodes
            if graph_data.get('nodes'):
                print("\n   Sample Nodes:")
                for node in graph_data['nodes'][:5]:
                    print(f"      - {node.get('name')} ({node.get('type')})")
            
            # Show sample edges
            if graph_data.get('edges'):
                print("\n   Sample Relationships:")
                for edge in graph_data['edges'][:5]:
                    print(f"      - {edge.get('source')} → {edge.get('type')} → {edge.get('target')}")
        else:
            print(f"❌ Error: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Check graph stats: GET http://localhost:5000/graph/stats")
    print("2. View graph: GET http://localhost:5000/graph/view")
    print("3. Ingest more documents: POST http://localhost:5000/graph/ingest-text")

if __name__ == "__main__":
    test_ingestion()

