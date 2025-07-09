from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import END, StateGraph

from graph.chains.hallucination_grader import hallucination_grader
from graph.consts import RETRIEVE, GRADE_DOCUMENTS, GENERATE, GENERATE_FALLBACK
from graph.nodes import generate, grade_documents, retrieve, generate_fallback
from graph.state import GraphState
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
    
def is_conversational_followup(question: str, chat_history: list) -> bool:
    """Check if the question is a conversational follow-up referencing previous conversation."""
    if not chat_history:
        return False
    
    # Define patterns that indicate conversational follow-ups
    followup_patterns = [
        "what did i just ask",
        "i don't understand",
        "can you clarify",
        "what do you mean",
        "explain again",
        "rephrase",
        "make it clearer",
        "i'm confused",
        "that doesn't make sense",
        "can you re-explain"
    ]
    
    question_lower = question.lower()
    return any(pattern in question_lower for pattern in followup_patterns)

def decide_to_generate(state):
    print("---ASSESS GRADED DOCUMENTS---")

    filtered_docs = state.get("documents", [])
    question = state.get("question", "")
    chat_history = state.get("chat_history", [])
    
    # Check if this is a conversational follow-up
    if is_conversational_followup(question, chat_history):
        print("---DECISION: CONVERSATIONAL FOLLOW-UP DETECTED, USE FALLBACK WITH CONTEXT---")
        return GENERATE_FALLBACK
    
    if filtered_docs:
        print("---DECISION: DOCUMENTS ARE RELEVANT, PROCEED TO GENERATE---")
        return GENERATE
    else:
        print("---DECISION: NO RELEVANT DOCUMENTS, END WORKFLOW---")
        return GENERATE_FALLBACK
    
def grade_generation_grounded_in_documents(state: GraphState) -> str:
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]    

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    if hallucination_grade := score.binary_score:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        return "supported"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-GENERATE ANSWERS---")
        return "not supported"

workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(GENERATE_FALLBACK, generate_fallback)

workflow.set_entry_point(RETRIEVE)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)

workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        GENERATE: GENERATE,
        GENERATE_FALLBACK: GENERATE_FALLBACK
    }
)

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents,
    {
        "not supported": GENERATE,  # Regenerate if not supported
        "supported": END,              # End if supported
    }
)

workflow.add_edge(GENERATE_FALLBACK, END)

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph.png")
    