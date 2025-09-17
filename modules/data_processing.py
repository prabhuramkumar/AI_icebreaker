"""Module for processing LinkedIn profile data."""

import json
import logging
from typing import Dict, List, Any, Optional

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

from modules.llm_interface import create_watsonx_embedding
import config

logger = logging.getLogger(__name__)

def split_profile_data(profile_data: Dict[str, Any]) -> List:
    """Splits the LinkedIn profile JSON data into nodes.
    
    Args:
        profile_data: LinkedIn profile data dictionary.
        
    Returns:
        List of document nodes.
    """
    try:
        # Convert the profile data to a JSON string
        profile_json = json.dumps(profile_data)

        # Create a Document object from the JSON string
        document = Document(text=profile_json)

        # Split the document into nodes using SentenceSplitter
        splitter = SentenceSplitter(chunk_size=config.CHUNK_SIZE)
        nodes = splitter.get_nodes_from_documents([document])
        
        logger.info(f"Created {len(nodes)} nodes from profile data")
        return nodes
    except Exception as e:
        logger.error(f"Error in split_profile_data: {e}")
        return []

def create_vector_database(nodes: List) -> Optional[VectorStoreIndex]:
    """Stores the document chunks (nodes) in a vector database.
    
    Args:
        nodes: List of document nodes to be indexed.
        
    Returns:
        VectorStoreIndex or None if indexing fails.
    """
    # TODO: Implement this function to create a vector database
    # 1. Get the embedding model using create_watsonx_embedding()
    # 2. Create a VectorStoreIndex from the nodes
    # 3. Return the index
    
    return None  # Replace with your implementation

def verify_embeddings(index: VectorStoreIndex) -> bool:
    """Verify that all nodes have been properly embedded.
    
    Args:
        index: VectorStoreIndex to verify.
        
    Returns:
        True if all embeddings are valid, False otherwise.
    """
    # TODO: Implement this function to verify embeddings
    # 1. Get the vector store from the index
    # 2. Get the node IDs from the index
    # 3. Check if each node has a valid embedding
    # 4. Return True if all embeddings are valid, False otherwise
    
    return False  # Replace with your implementation