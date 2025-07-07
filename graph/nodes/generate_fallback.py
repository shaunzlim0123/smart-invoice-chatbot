from typing import Any, Dict

from graph.chains.fallback_generation import fallback_generation_chain
from graph.state import GraphState
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def generate_fallback(state: GraphState) -> Dict[str, Any]:
    """
    Generate answer without documents (fallback)
    """
    print("---GENERATE FALLBACK ANSWER---")
    question = state["question"]
    chat_history = state.get("chat_history", [])
    
    generation = fallback_generation_chain.invoke({
        "question": question,
        "chat_history": chat_history
    })
    
    # Generate follow-up questions for fallback
    follow_up_template = """Based on the question '{question}' and the answer '{answer}', 
    suggest 3 relevant follow-up questions that would help the user explore this topic further.
    Format them as a numbered list."""
    
    follow_up_prompt = PromptTemplate(
        template=follow_up_template, 
        input_variables=["question", "answer"]
    )
    follow_up_chain = LLMChain(llm=chat, prompt=follow_up_prompt)
    follow_up_questions = follow_up_chain.run(
        question=question, 
        answer=generation
    )
    
    return {
        "question": question, 
        "generation": generation,
        "follow_up_questions": follow_up_questions
    }
