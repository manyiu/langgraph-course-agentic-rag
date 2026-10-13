from dotenv import load_dotenv

from langgraph_course_agentic_rag.graph.chains.generation import generation_chain
from langgraph_course_agentic_rag.graph.chains.hallucination_grader import (
    GradeHallucinations,
    hallucination_grader,
)
from langgraph_course_agentic_rag.graph.chains.retrieval_grader import (
    GradeDocument,
    retrieval_grader,
)
from langgraph_course_agentic_rag.graph.chains.router import RouteQuery, question_router
from langgraph_course_agentic_rag.ingestion import retriever

load_dotenv()

def test_retrieval_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_text = docs[0].page_content

    res: GradeDocument = retrieval_grader.invoke({"document": doc_text, "question": question})
    assert res.binary_score == "yes"

def test_retrieval_grader_answer_no() -> None:
    question = "how to make pizza"
    docs = retriever.invoke(question)
    doc_text = docs[0].page_content

    res: GradeDocument = retrieval_grader.invoke({"document": doc_text, "question": question})
    assert res.binary_score == "no"

def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({
        "context": docs,
        "question": question,
    })

def test_hallucination_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
   
    generation = generation_chain.invoke({
        "context": docs,
        "question": question,
    })

    res: GradeHallucinations = hallucination_grader.invoke({
        "documents": docs,
        "generation": generation,
    })
    assert res.binary_score

def test_hallucination_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    res: GradeHallucinations = hallucination_grader.invoke({
        "documents": docs,
        "generation": "In order to make pizza, you need to follow these steps: 1. Mix the dough, 2. Bake the pizza, 3. Add the toppings.",
    })
    assert not res.binary_score

def test_router_to_vectorstore() -> None:
    question = "agent memory"
    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "vectorstore"

def test_router_to_websearch() -> None:
    question = "how to make pizza"
    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "websearch"