from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents"""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

structured_llm_grader = llm.with_structured_output(GradeDocuments)

system = """You are a grader assessing whether a retrieved document helps answer a user’s question.

- Look for direct matches on key terms and broader semantic connections.
- Mark 'yes' if the document provides at least partial, concrete support for answering the question.
- Mark 'no' if it offers no meaningful or on-topic information.
- Only respond with “yes” or “no”.
"""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}")
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader