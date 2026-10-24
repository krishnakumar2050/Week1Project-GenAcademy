"""LangGraph wiring only; domain behavior lives in specialist/report modules."""

from langgraph.graph import END, START, StateGraph

from coding_review.agents.prompts import SPECIALTIES
from coding_review.agents.reviewer import make_reviewer
from coding_review.backends.base import ReviewBackend
from coding_review.domain import State
from coding_review.reporting.aggregation import orchestrate


def build_graph(model: ReviewBackend):
    graph = StateGraph(State)
    for name in SPECIALTIES:
        graph.add_node(name, make_reviewer(name, model))
        graph.add_edge(START, name)
    graph.add_node("orchestrator", orchestrate)
    # A list of sources is an explicit barrier: all three must finish first.
    graph.add_edge(list(SPECIALTIES), "orchestrator")
    graph.add_edge("orchestrator", END)
    return graph.compile()
