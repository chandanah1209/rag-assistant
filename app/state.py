from typing import TypedDict, List, Optional

class GraphState(TypedDict):
    """
    State that flows between all nodes in the LangGraph workflow.
    """
    question: str
    rewritten_query: Optional[str]
    documents: List[dict]
    relevant_docs: List[dict]
    answer: Optional[str]
    retry_count: int
    query_type: Optional[str]