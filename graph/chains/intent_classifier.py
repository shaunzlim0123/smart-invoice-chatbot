from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class IntentClassification(BaseModel):
    """Intent classification for user questions"""

    intent: str = Field(
        description="The intent of the user's question: 'new_question' or 'clarification'"
    )
    rephrased_question: str = Field(
        description="If clarification, provide a standalone question that captures the clarification intent. If new question, return the original question."
    )

structured_llm_classifier = llm.with_structured_output(IntentClassification)

system = """You are an intent classifier that determines whether a user is asking a new question or asking for clarification about a previous question.

INTENT TYPES:
- 'new_question': User is asking a completely new question with no reference to previous context
- 'clarification': User is asking for clarification, follow-up, or additional details about a previous question

REPHRASED QUESTION:
- If 'new_question': Return the original question as-is
- If 'clarification': Create a standalone question that captures the clarification intent, making it self-contained and clear

Examples:
- "What is the weather like?" -> new_question, "What is the weather like?"
- "Can you tell me more about that?" -> clarification, "What are the key benefits and features of smart contracts?"
- "How does that work?" -> clarification, "How does blockchain technology work and what are its core principles?"
- "What are the risks?" -> clarification, "What are the potential risks and challenges of blockchain technology?"
- "What question did I just ask?" -> clarification, "What was the previous question I asked about blockchain technology?"
"""

classify_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Chat history: {chat_history}\n\nCurrent question: {question}")
    ]
)

intent_classifier = classify_prompt | structured_llm_classifier 