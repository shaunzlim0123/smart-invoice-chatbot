# from dotenv import load_dotenv

# load_dotenv()

# from graph.graph import app

# if __name__ == "__main__":
#     print("Hello Agentic RAG")
#     result = app.invoke(input={"question": "How do i make pizza?"})

#     print(result["generation"])


from dotenv import load_dotenv
load_dotenv()

from graph.graph import app
from langchain_core.messages import HumanMessage, AIMessage

def run_interactive_chat():
    """Run an interactive CLI chat session"""
    print("🤖 Welcome to Agentic RAG Chatbot!")
    print("Type 'quit' to exit.\n")
    
    chat_history = []
    
    while True:
        # Get user input
        query = input("\nYou: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not query:
            continue
        
        # Run the query
        print("\nProcessing your query...")
        result = app.invoke(input={
            "question": query,
            "chat_history": chat_history
        })
        
        # Update chat history
        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=result["generation"]))
        
        # Display the answer
        print(f"\n🤖 Assistant: {result['generation']}")
        
        # Display follow-up questions if available
        if result.get("follow_up_questions"):
            print(f"\n💡 Suggested follow-up questions:\n{result['follow_up_questions']}")

if __name__ == "__main__":
    # Example usage
    print("Running example query...")
    result = app.invoke(input={
        "question": "How do I make pizza?",
        "chat_history": []
    })
    print(f"Answer: {result['generation']}")
    print(f"\nFollow-up questions:\n{result.get('follow_up_questions', 'None')}")
    
    # Start interactive chat
    print("\n" + "="*50 + "\n")
    run_interactive_chat()