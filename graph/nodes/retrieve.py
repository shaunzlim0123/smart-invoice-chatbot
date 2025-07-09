from typing import Any, Dict

from graph.state import GraphState
from ingestion import chroma_retriever

def retrieve(state: GraphState) -> Dict[str, Any]:
    print("---RETRIEVE FROM CHROMA---")
    question = state["question"]
    
    # Use simple retriever
    documents = chroma_retriever.invoke(question)
    
    return {
        "documents": documents,
        "question": question
    }