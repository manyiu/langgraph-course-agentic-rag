from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

from langgraph_course_agentic_rag.graph.chains.answer_grader import answer_grader
from langgraph_course_agentic_rag.graph.chains.hallucination_grader import (
    hallucination_grader,
)
from langgraph_course_agentic_rag.graph.chains.router import RouteQuery, question_router
from langgraph_course_agentic_rag.graph.consts import (
    GENERATE,
    GRADE_DOCUMENTS,
    RETRIEVE,
    WEB_SEARCH,
)
from langgraph_course_agentic_rag.graph.nodes import (
    generate,
    grade_documents,
    retrieve,
    web_search,
)
from langgraph_course_agentic_rag.graph.state import GraphState

load_dotenv()


def decide_to_generate(state: GraphState) -> str:
    if state["web_search"]:
        return WEB_SEARCH
    else:
        return GENERATE


def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    question = state["question"]
    generation = state["generation"]
    documents = state["documents"]
    score = hallucination_grader.invoke(
        {
            "documents": documents,
            "generation": generation,
        }
    )
    if hallucination_grade := score.binary_score:
        score = answer_grader.invoke(
            {
                "question": question,
                "generation": generation,
            }
        )
        if answer_grade := score.binary_score:
            return "useful"
        else:
            return "not useful"
    else:
        return "not supported"

def route_question(state: GraphState) -> str:
    question = state["question"]
    res: RouteQuery = question_router.invoke({"question": question})
    
    if res.datasource == "vectorstore":
        return RETRIEVE
    else:
        return WEB_SEARCH


workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEB_SEARCH, web_search)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)

workflow.set_conditional_entry_point(
    route_question,
    {
        RETRIEVE: RETRIEVE,
        WEB_SEARCH: WEB_SEARCH,
    },
)

workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        WEB_SEARCH: WEB_SEARCH,
        GENERATE: GENERATE,
    },
)
workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "not supported": GENERATE,
        "useful": END,
        "not useful": WEB_SEARCH,
    },
)


workflow.add_edge(WEB_SEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

app = workflow.compile()
app.get_graph().draw_mermaid_png(output_file_path="rag_workflow.png")
