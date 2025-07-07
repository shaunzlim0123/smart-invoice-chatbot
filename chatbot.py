from dotenv import load_dotenv
load_dotenv()

from graph.graph import app
from langchain_core.messages import HumanMessage, AIMessage
import streamlit as st
from typing import Set

class RAGChatbot:
    def __init__(self):
        self.app = app
        self.chat_history = []
    
    def add_to_history(self, human_msg: str, ai_msg: str):
        """Add messages to chat history"""
        self.chat_history.append(HumanMessage(content=human_msg))
        self.chat_history.append(AIMessage(content=ai_msg))
    
    def run_query(self, query: str):
        """Run a query through the LangGraph workflow"""
        result = self.app.invoke(input={
            "question": query,
            "chat_history": self.chat_history
        })
        
        # Add to history
        self.add_to_history(query, result["generation"])
        
        return {
            "query": query,
            "result": result["generation"],
            "source_documents": result.get("documents", []),
            "follow_up_questions": result.get("follow_up_questions", "")
        }

def create_sources_string(documents) -> str:
    """Create a formatted string of source documents"""
    if not documents:
        return ""
    
    sources_string = "**Sources:**\n"
    seen_sources = set()
    
    for i, doc in enumerate(documents):
        # Extract source info from metadata if available
        source = doc.metadata.get("source", f"Document {i+1}")
        page = doc.metadata.get("page", None)
        
        if source not in seen_sources:
            seen_sources.add(source)
            if page:
                sources_string += f"- {source} (Page {page})\n"
            else:
                sources_string += f"- {source}\n"
    
    return sources_string

def process_followup_question(question: str):
    """Process a follow-up question and add it to the chat"""
    # Add user message
    st.session_state.messages.append({"role": "user", "content": question})
    
    # Get bot response
    with st.spinner("Thinking..."):
        response = st.session_state.chatbot.run_query(question)
    
    # Create sources string
    sources_str = create_sources_string(response["source_documents"])
    
    # Add assistant message to history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response["result"],
        "follow_up": response["follow_up_questions"],
        "sources": sources_str
    })

# Streamlit UI
def main():
    st.title("🤖 Agentic RAG Chatbot")
    st.markdown("Ask me anything! I'll retrieve relevant documents and provide grounded answers.")
    
    # Initialize session state
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = RAGChatbot()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show source documents for assistant messages
            if message["role"] == "assistant" and "sources" in message and message["sources"]:
                with st.expander("📚 View Sources"):
                    st.markdown(message["sources"])
            
            # Show clickable follow-up questions
            if message["role"] == "assistant" and "follow_up" in message and message["follow_up"]:
                st.markdown("**💡 Suggested follow-up questions:**")
                
                # Parse follow-up questions
                questions = message["follow_up"].split('\n')
                
                for q_idx, question in enumerate(questions):
                    if question.strip():
                        # Clean the question (remove numbering)
                        clean_q = question.strip()
                        if clean_q and len(clean_q) > 2:
                            if clean_q[0].isdigit() and clean_q[1:3] in ['. ', '- ', ') ']:
                                clean_q = clean_q[3:].strip()
                            elif clean_q[0].isdigit() and clean_q[1] in ['.', '-', ')']:
                                clean_q = clean_q[2:].strip()
                        
                        if clean_q:  # Only create button if question is not empty
                            if st.button(
                                f"❓ {clean_q}",
                                key=f"follow_up_{idx}_{q_idx}",
                                use_container_width=True
                            ):
                                process_followup_question(clean_q)
                                st.rerun()
    
    # Chat input
    prompt = st.chat_input("What would you like to know?")
    
    # Process regular prompt
    if prompt:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chatbot.run_query(prompt)
            
            st.markdown(response["result"])
            
            # Create sources string
            sources_str = create_sources_string(response["source_documents"])
            
            # Show source documents if available
            if sources_str:
                with st.expander("📚 View Sources"):
                    st.markdown(sources_str)
            
            # Show clickable follow-up questions immediately
            if response["follow_up_questions"]:
                st.markdown("**💡 Suggested follow-up questions:**")
                
                questions = response["follow_up_questions"].split('\n')
                current_msg_idx = len(st.session_state.messages)
                
                for q_idx, question in enumerate(questions):
                    if question.strip():
                        # Clean the question
                        clean_q = question.strip()
                        if clean_q and len(clean_q) > 2:
                            if clean_q[0].isdigit() and clean_q[1:3] in ['. ', '- ', ') ']:
                                clean_q = clean_q[3:].strip()
                            elif clean_q[0].isdigit() and clean_q[1] in ['.', '-', ')']:
                                clean_q = clean_q[2:].strip()
                        
                        if clean_q:
                            if st.button(
                                f"❓ {clean_q}",
                                key=f"follow_up_new_{current_msg_idx}_{q_idx}",
                                use_container_width=True
                            ):
                                process_followup_question(clean_q)
                                st.rerun()
            
            # Add to message history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response["result"],
                "follow_up": response["follow_up_questions"],
                "sources": sources_str
            })

if __name__ == "__main__":
    main()