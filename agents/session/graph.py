from __future__ import annotations

from langgraph.graph import END, StateGraph

from agents.session.nodes import (
    append_working_memory_node, generate_node, inject_node, load_working_memory_node,
    retrieve_episodic_node, retrieve_long_term_node,
)
from agents.session.state import SessionState


def build_session_graph():
    graph = StateGraph(SessionState)
    graph.add_node("load_working", load_working_memory_node)
    graph.add_node("retrieve_episodic", retrieve_episodic_node)
    graph.add_node("retrieve_long_term", retrieve_long_term_node)
    graph.add_node("inject", inject_node)
    graph.add_node("generate", generate_node)
    graph.add_node("append_working", append_working_memory_node)

    graph.set_entry_point("load_working")
    # Fan out: episodic and long_term run in parallel after working memory loads
    graph.add_edge("load_working", "retrieve_episodic")
    graph.add_edge("load_working", "retrieve_long_term")
    # Both feed into inject once complete
    graph.add_edge("retrieve_episodic", "inject")
    graph.add_edge("retrieve_long_term", "inject")
    graph.add_edge("inject", "generate")
    # Log this turn to working memory once generation completes (or no-ops if
    # there was no user_message, e.g. a bare session_init call).
    graph.add_edge("generate", "append_working")
    graph.add_edge("append_working", END)

    return graph.compile()


session_graph = build_session_graph()
