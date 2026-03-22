from typing import TypedDict, List, Optional

class AgentState(TypedDict):
    """
    The state that gets passed between every node in the agent graph.
    Think of it as the agent's working memory at each step.
    """
    # The user's original question
    question: str
    
    # Retrieved chunks from ChromaDB
    retrieved_chunks: List[str]
    
    # The final generated answer
    answer: str
    
    # What the agent decided to do
    decision: str
    
    # How many times the agent has tried to answer
    iterations: int
    
    # Conversation history
    chat_history: List[dict]
    
    # Whether the retrieval found relevant content
    retrieval_confidence: float