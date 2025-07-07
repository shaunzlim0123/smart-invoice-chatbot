from typing import List, TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import operator

class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    question: str
    chat_history: Annotated[Sequence[BaseMessage], operator.add]
    generation: str
    documents: List[str]
    follow_up_questions: str