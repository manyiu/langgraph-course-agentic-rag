from typing import Any, Dict

from langgraph_course_agentic_rag.graph.state import GraphState
from langgraph_course_agentic_rag.ingestion import retriever


def retrieve(state: GraphState) -> Dict[str, Any]:
    question = state["question"]

    documents = retriever.invoke(question)

    return {"documents": documents, "question": question}