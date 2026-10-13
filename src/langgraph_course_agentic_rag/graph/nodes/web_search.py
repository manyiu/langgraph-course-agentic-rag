from typing import Any, Dict
from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from langgraph_course_agentic_rag.graph.state import GraphState

web_search_tool = TavilySearch(max_results=3)

def web_search(state: GraphState) -> Dict[str, Any]:
    question = state["question"]
    documents = state.get("documents")

    tavily_results = web_search_tool.invoke({"query": question})
    joined_tavily_result = "\n\n".join([r["content"] for r in tavily_results])
    web_results = Document(page_content=joined_tavily_result)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]

    return {"documents": documents, "question": question}