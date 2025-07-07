from typing import Any, Dict

from graph.state import GraphState
from ingestion import history_aware_retriever

def retrieve(state: GraphState) -> Dict[str, Any]:
    print("---RETRIEVE WITH HISTORY AWARENESS FROM CHROMA---")
    question = state["question"]
    chat_history = state.get("chat_history", [])
    
    # Use history-aware retriever
    documents = history_aware_retriever.invoke({
        "input": question,
        "chat_history": chat_history
    })
    
    return {"documents": documents, "question": question}