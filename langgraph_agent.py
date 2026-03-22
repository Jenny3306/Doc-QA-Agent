from langgraph.graph import StateGraph, END
from agent_state import AgentState
from agent_nodes import (
    router_node,
    retriever_node,
    generator_node,
    meta_node,
    clarifier_node
)
import warnings
warnings.filterwarnings("ignore")


def route_decision(state: AgentState) -> str:
    """
    Conditional edge — tells LangGraph which node to go to next
    based on the router's decision.
    """
    decision = state["decision"]
    
    if decision == "retrieve":
        return "retriever"
    elif decision == "meta":
        return "meta"
    elif decision == "clarify":
        return "clarifier"
    else:
        return "retriever"  # Default fallback


def build_agent():
    """Build and compile the LangGraph agent."""
    
    # Create the graph
    graph = StateGraph(AgentState)
    
    # Add all nodes
    graph.add_node("router", router_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("generator", generator_node)
    graph.add_node("meta", meta_node)
    graph.add_node("clarifier", clarifier_node)
    
    # Set entry point
    graph.set_entry_point("router")
    
    # Add conditional edges from router
    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "retriever": "retriever",
            "meta": "meta",
            "clarifier": "clarifier"
        }
    )
    
    # After retrieval always generate
    graph.add_edge("retriever", "generator")
    
    # All paths end after generating/handling
    graph.add_edge("generator", END)
    graph.add_edge("meta", END)
    graph.add_edge("clarifier", END)
    
    return graph.compile()


def run_agent():
    """Run the full interactive agent."""
    
    agent = build_agent()
    chat_history = []
    
    print("\n" + "=" * 55)
    print("  NVIDIA Document Agent — LangGraph Edition")
    print("  Powered by Nemotron + ChromaDB + LangGraph")
    print("=" * 55)
    print("  Commands: 'quit' | 'clear' | 'history'")
    print("=" * 55 + "\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() == "quit":
            print("Goodbye! Great work today.")
            break
        
        if user_input.lower() == "clear":
            chat_history = []
            print("Memory cleared!\n")
            continue
        
        if user_input.lower() == "history":
            if not chat_history:
                print("No history yet.\n")
            else:
                print(f"\n{len(chat_history)//2} exchanges in memory:")
                for i, msg in enumerate(chat_history):
                    role = "You" if msg["role"] == "user" else "Agent"
                    print(f"  {role}: {msg['content'][:80]}...")
                print()
            continue
        
        if not user_input:
            continue
        
        # Build initial state
        initial_state: AgentState = {
            "question": user_input,
            "retrieved_chunks": [],
            "answer": "",
            "decision": "",
            "iterations": 0,
            "chat_history": chat_history,
            "retrieval_confidence": 0.0
        }
        
        # Run the agent graph
        print()
        final_state = agent.invoke(initial_state)
        answer = final_state["answer"]
        decision = final_state["decision"]
        
        # Update chat history
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": answer})
        
        # Keep history manageable
        if len(chat_history) > 12:
            chat_history = chat_history[-12:]
        
        print(f"Agent [{decision.upper()}]: {answer}\n")


if __name__ == "__main__":
    run_agent()