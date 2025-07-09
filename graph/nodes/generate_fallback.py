from typing import Any, Dict

from graph.chains.fallback_generation import fallback_generation_chain
from graph.state import GraphState
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def generate_fallback(state: GraphState) -> Dict[str, Any]:
    """
    Generate answer without documents (fallback)
    """
    print("---GENERATE FALLBACK ANSWER---")
    question = state["question"]
    
    # Generate answer using just the question, without chat history
    generation = fallback_generation_chain.invoke({
        "question": question
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
        "question": question, 
        "generation": generation,
        "follow_up_questions": follow_up_questions
    }