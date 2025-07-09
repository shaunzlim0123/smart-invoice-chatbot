from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import END, StateGraph


from graph.chains.hallucination_grader import hallucination_grader
from graph.consts import RETRIEVE, GRADE_DOCUMENTS, GENERATE, GENERATE_FALLBACK
from graph.nodes import generate, grade_documents, retrieve, generate_fallback
from graph.state import GraphState
    
def decide_to_generate(state):
    print("---ASSESS GRADED DOCUMENTS---")

    filtered_docs = state.get("documents", [])
    
    if filtered_docs:
        print("---DECISION: DOCUMENTS ARE RELEVANT, PROCEED TO GENERATE---")
        return GENERATE
    else:
        print("---DECISION: NO RELEVANT DOCUMENTS, USE FALLBACK---")
        return GENERATE_FALLBACK
    
def grade_generation_grounded_in_documents(state: GraphState) -> str:
    print("---CHECK HALLUCINATIONS---")
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
    