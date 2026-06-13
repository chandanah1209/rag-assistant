from langgraph.graph import StateGraph, END
from app.state import GraphState
from app.nodes import (
    query_analysis_node,
    retrieval_node,
    grading_node,
    generation_node,
    no_answer_node
)

MAX_RETRIES = 2

def route_after_grading(state: GraphState) -> str:
    """Decide what to do based on grading results"""
    if len(state["relevant_docs"]) > 0:
        return "generate"
    elif state["retry_count"] < MAX_RETRIES:
        return "retry"
    else:
        return "no_answer"

def rewrite_retry_node(state: GraphState) -> GraphState:
    """Rewrite query and increment retry count"""
    new_query = state["rewritten_query"] + " explain with example"
    return {
        **state,
        "rewritten_query": new_query,
        "retry_count": state["retry_count"] + 1
    }

def build_graph():
    """Build and return the compiled LangGraph workflow"""
    workflow = StateGraph(GraphState)

    # Add all nodes
    workflow.add_node("query_analysis", query_analysis_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("grading", grading_node)
    workflow.add_node("generation", generation_node)
    workflow.add_node("retry", rewrite_retry_node)
    workflow.add_node("no_answer", no_answer_node)

    # Set entry point
    workflow.set_entry_point("query_analysis")

    # Add edges
    workflow.add_edge("query_analysis", "retrieval")
    workflow.add_edge("retrieval", "grading")

    # Conditional edge — self corrective part
    workflow.add_conditional_edges(
        "grading",
        route_after_grading,
        {
            "generate": "generation",
            "retry": "retry",
            "no_answer": "no_answer"
        }
    )

    # Retry loops back to retrieval
    workflow.add_edge("retry", "retrieval")

    # End points
    workflow.add_edge("generation", END)
    workflow.add_edge("no_answer", END)

    return workflow.compile()