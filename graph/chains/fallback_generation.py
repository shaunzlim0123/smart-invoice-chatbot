from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Fallback generation prompt
fallback_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant for SmartFund applications. 
    
    The user has asked a question that is not covered in the available SmartFund documentation. 
    
    Your task is to:
    1. Politely explain that the question is outside the scope of the SmartFund documentation
    2. Suggest what topics ARE covered in the SmartFund documentation (Smart Invoice app, EMI features, user manual topics, FAQs)
    3. Encourage the user to rephrase their question or contact support
    
    Be helpful, professional, and maintain a friendly tone."""),
    ("human", "Question: {question}")
])

# Create the fallback generation chain
fallback_generation_chain = (
    fallback_prompt
    | ChatOpenAI(model="gpt-4o-mini", temperature=0)
    | StrOutputParser()
)