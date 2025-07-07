from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)


rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant for question-answering tasks."),
    ("human", '''Use the retrieved context to answer the question appropriately:
    - If the question begins with “How” (i.e., it’s a process or procedural query), respond with explicit numbered steps (1., 2., 3., …), each no more than two sentences.
    - Otherwise (for simple fact-based or conceptual questions), respond with a single concise paragraph explanation (no more than five sentences).
    - If you don’t know the answer, simply say “I don’t know.”
    - Include concrete examples when they help illustrate your point.

    Question: {question}
    Context: {context}
    Answer:''')
])


generation_chain = rag_prompt | llm | StrOutputParser()
