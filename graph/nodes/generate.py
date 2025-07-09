from typing import Any, Dict

from graph.chains.generation import generation_chain
from graph.state import GraphState
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def generate(state: GraphState) -> dict:
    """
    Generate answer using retrieved documents
    """
    print("---GENERATE ANSWER---")
    question = state["question"]
    documents = state["documents"]
    chat_history = state.get("chat_history", [])
    
    # Generate answer with context
    generation = generation_chain.invoke({
        "context": documents, 
        "question": question,
        "chat_history": chat_history
    })
    
    # Generate follow-up questions
    follow_up_template = """Based on the question '{question}' and the answer '{answer}', 
    suggest 3 relevant follow-up questions that would help the user explore this topic further.
    Format them as a numbered list."""
    
    follow_up_prompt = PromptTemplate(
        template=follow_up_template, 
        input_variables=["question", "answer"]
    )
    follow_up_chain = follow_up_prompt | chat

    follow_up_response = follow_up_chain.invoke({
        "question": question, 
        "answer": generation
    })

    follow_up_questions = follow_up_response.content
    
    return {
        "documents": documents, 
        "question": question, 
        "generation": generation,
        "follow_up_questions": follow_up_questions
    }