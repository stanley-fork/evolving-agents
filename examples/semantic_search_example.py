# examples/semantic_search_example.py

import asyncio
import sys
import os
import json
import logging
from typing import Dict, Any, List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evolving_agents.smart_library.smart_library import SmartLibrary
from evolving_agents.utils.embeddings import embedding_service

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_record_summary(record: Dict[str, Any]):
    """Print a summary of a record."""
    print(f"Name: {record.get('name')}")
    print(f"Type: {record.get('record_type')}")
    print(f"Domain: {record.get('domain')}")
    print(f"Description: {record.get('description')}")
    
    if "tags" in record and record["tags"]:
        print(f"Tags: {', '.join(record['tags'])}")
    
    if "metadata" in record and record["metadata"]:
        print("Metadata:")
        for key, value in record["metadata"].items():
            print(f"  {key}: {value}")
    
    print("-" * 40)

async def run_semantic_search():
    try:
        # Load the library
        library = SmartLibrary("simplified_agent_library.json")
        
        print(f"Library loaded with {len(library.records)} records")
        
        # Define search queries
        search_queries = [
            "document analysis specialist",
            "invoice processing agent",
            "medical record analysis",
            "agent that can coordinate between other agents",
            "communication between agents"
        ]
        
        for query in search_queries:
            print("\n" + "="*80)
            print(f"Searching for: '{query}'")
            print("="*80)
            
            # Perform semantic search
            results = await library.semantic_search(
                query=query,
                limit=3,
                threshold=0.3  # Only return results with similarity above 0.3
            )
            
            if results:
                print(f"Found {len(results)} results:")
                for i, (record, score) in enumerate(results):
                    print(f"\nResult {i+1} - Similarity Score: {score:.4f}")
                    print_record_summary(record)
            else:
                print("No matching records found")
        
        # Test searching within a specific domain
        domain = "document_processing"
        query = "specialist for analyzing documents"
        
        print("\n" + "="*80)
        print(f"Searching in domain '{domain}' for: '{query}'")
        print("="*80)
        
        domain_results = await library.semantic_search(
            query=query,
            domain=domain,
            limit=3
        )
        
        if domain_results:
            print(f"Found {len(domain_results)} results in domain '{domain}':")
            for i, (record, score) in enumerate(domain_results):
                print(f"\nResult {i+1} - Similarity Score: {score:.4f}")
                print_record_summary(record)
        else:
            print(f"No matching records found in domain '{domain}'")
        
        # Test searching by record type
        record_type = "TOOL"
        query = "tool for agent communication"
        
        print("\n" + "="*80)
        print(f"Searching for {record_type}s matching: '{query}'")
        print("="*80)
        
        type_results = await library.semantic_search(
            query=query,
            record_type=record_type,
            limit=3
        )
        
        if type_results:
            print(f"Found {len(type_results)} {record_type}s:")
            for i, (record, score) in enumerate(type_results):
                print(f"\nResult {i+1} - Similarity Score: {score:.4f}")
                print_record_summary(record)
        else:
            print(f"No matching {record_type}s found")
        
    except Exception as e:
        logger.error(f"Error in semantic search: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_semantic_search())