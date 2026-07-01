from __future__ import annotations

from langgraph.graph import END, StateGraph

from agents.retrieval.nodes import (
    assemble_node, embed_query_node, expand_query_node,
    fuse_node, hybrid_search_node, rerank_node,
)
from agents.retrieval.state import RetrievalState


def build_retrieval_graph():
    graph = StateGraph(RetrievalState)
    graph.add_node("expand_query", expand_query_node)
    graph.add_node("embed_query", embed_query_node)
    graph.add_node("hybrid_search", hybrid_search_node)
    graph.add_node("fuse", fuse_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("assemble", assemble_node)

    graph.set_entry_point("expand_query")
    graph.add_edge("expand_query", "embed_query")
    graph.add_edge("embed_query", "hybrid_search")
    graph.add_edge("hybrid_search", "fuse")
    graph.add_edge("fuse", "rerank")
    graph.add_edge("rerank", "assemble")
    graph.add_edge("assemble", END)

    return graph.compile()


retrieval_graph = build_retrieval_graph()
