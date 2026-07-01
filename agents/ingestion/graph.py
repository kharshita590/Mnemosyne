from __future__ import annotations

from langgraph.graph import END, StateGraph

from agents.ingestion.nodes import (
    chunk_node, classify_node, embed_node, extract_entities_node,
    extract_node, store_facts_node, store_node,
)
from agents.ingestion.state import IngestionState


def build_ingestion_graph():
    graph = StateGraph(IngestionState)
    graph.add_node("extract", extract_node)
    graph.add_node("extract_entities", extract_entities_node)
    graph.add_node("classify", classify_node)
    graph.add_node("chunk", chunk_node)
    graph.add_node("embed", embed_node)
    graph.add_node("store", store_node)
    graph.add_node("store_facts", store_facts_node)

    graph.set_entry_point("extract")
    # extract_node and extract_entities_node run in parallel
    graph.add_edge("extract", "extract_entities")
    graph.add_edge("extract", "classify")
    # Both feed into chunk once complete
    graph.add_edge("extract_entities", "chunk")
    graph.add_edge("classify", "chunk")
    graph.add_edge("chunk", "embed")
    graph.add_edge("embed", "store")
    graph.add_edge("store", "store_facts")
    graph.add_edge("store_facts", END)

    return graph.compile()


ingestion_graph = build_ingestion_graph()
