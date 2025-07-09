from typing import Any, Dict

from graph.chains.fallback_generation import fallback_generation_chain
from graph.state import GraphState
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def rewrite_question_with_context(question: str, chat_history: list) -> str:
    """Rewrite a question to include context from chat history when it's referencing previous conversation."""
    if not chat_history:
        return question
    
    # Create a prompt to rewrite the question with context
    context_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are helping to rewrite a user's question to include context from their chat history.
        
        The user has asked a question that seems to reference previous conversation (like "what did I just ask" or "I don't understand").
        
        Your task is to rewrite the question to include the relevant context from the chat history so it can be understood standalone.
        
        If the question is asking for clarification about a previous response, include the key parts of that response in the rewritten question.
        If the question is asking about a previous question, include that question in the rewritten question.
        
        Keep the rewritten question clear and focused.
        
        Chat History: {chat_history}"""),
        ("human", "Original question: {question}\n\nRewrite this question to include necessary context:")
    ])
    
    context_chain = context_prompt | chat
    
    try:
        result = context_chain.invoke({
            "question": question,
            "chat_history": chat_history
        })
        rewritten_question = result.content
        return rewritten_question
    except Exception:
        return question

def generate_fallback(state: GraphState) -> Dict[str, Any]:
    """
    Generate answer without documents (fallback)
    """
    print("---GENERATE FALLBACK ANSWER---")
    question = state["question"]
    chat_history = state.get("chat_history", [])
    
    # Check if we need to rewrite the question with context
    rewritten_question = rewrite_question_with_context(question, chat_history)
    
    # Use the rewritten question for better context
    generation = fallback_generation_chain.invoke({
        "question": rewritten_question,
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
