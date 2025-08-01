from typing import Any, Dict
from graph.chains.intent_classifier import intent_classifier
from graph.state import GraphState

def classify_intent(state: GraphState) -> Dict[str, Any]:
    """
    Classifies the user's intent and rephrases the question if needed.
    
    Args:
        state (GraphState): The current graph state containing question and chat history
    
    Returns:
        Dict[str, Any]: Updated state with processed question and intent information
    """
    
    print("---CLASSIFYING USER INTENT---")
    question = state["question"]
    chat_history = state.get("chat_history", [])
    
    # Format chat history for the classifier
    formatted_history = ""
    if chat_history:
        formatted_history = "\n".join([f"Q: {q}\nA: {a}" for q, a in chat_history])
    
    # Classify the intent
    result = intent_classifier.invoke({
        "question": question,
        "chat_history": formatted_history
    })
    
    intent = result.intent
    rephrased_question = result.rephrased_question
    
    print(f"---INTENT: {intent.upper()}---")
    if intent == "clarification":
        print(f"---REPHRASED QUESTION: {rephrased_question}---")
    else:
        print("---NEW QUESTION DETECTED---")
    
    return {
        "question": rephrased_question,
        "original_question": question,
        "intent": intent,
        "chat_history": chat_history
    } 