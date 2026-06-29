from typing import TypedDict, List
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from retriever import retrieve_chunks, load_existing_vectorstore
from critic import critique_answer
from dotenv import load_dotenv
import os

load_dotenv()

class RAGState(TypedDict):
    question: str
    context: List[Document]
    answer: str
    verdict: str
    reason: str
    reformulated_query: str
    retry_count: int
    final_answer: str
    trace_log: List[str]

MAX_RETRIES = 3

def get_llm():
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0.2
    )

def retrieve_node(state: RAGState) -> RAGState:
    query = state.get("reformulated_query") or state["question"]
    vectorstore = load_existing_vectorstore()
    docs = retrieve_chunks(query, vectorstore, k=5)
    log = state.get("trace_log", [])
    log.append(f"Retrieved {len(docs)} chunks for query: '{query}'")
    return {**state, "context": docs, "trace_log": log}

def generate_node(state: RAGState) -> RAGState:
    context_text = "\n\n".join([doc.page_content for doc in state["context"]])

    prompt = PromptTemplate(
        input_variables=["question", "context"],
        template="""
You are a helpful assistant. Answer the question using ONLY the provided context.
If the context does not contain enough information, say exactly:
"I don't have enough information in the provided documents to answer this question."

CONTEXT:
{context}

QUESTION: {question}

ANSWER:
"""
    )

    llm = get_llm()
    chain = prompt | llm
    response = chain.invoke({
        "question": state["question"],
        "context": context_text
    })

    log = state.get("trace_log", [])
    log.append(f"Generated answer (attempt {state.get('retry_count', 0) + 1})")
    return {**state, "answer": response.content.strip(), "trace_log": log}

def critic_node(state: RAGState) -> RAGState:
    result = critique_answer(
        question=state["question"],
        answer=state["answer"],
        context_chunks=state["context"]
    )

    log = state.get("trace_log", [])
    log.append(f"Critic verdict: {result['verdict']} — {result['reason']}")

    return {
        **state,
        "verdict": result["verdict"],
        "reason": result["reason"],
        "reformulated_query": result["reformulated_query"],
        "trace_log": log
    }

def decision_node(state: RAGState) -> str:
    retry_count = state.get("retry_count", 0)

    if state["verdict"] == "PASS":
        return "accept"
    elif retry_count >= MAX_RETRIES:
        return "give_up"
    else:
        return "retry"

def accept_node(state: RAGState) -> RAGState:
    log = state.get("trace_log", [])
    log.append("Answer accepted by critic.")
    return {**state, "final_answer": state["answer"], "trace_log": log}

def retry_node(state: RAGState) -> RAGState:
    log = state.get("trace_log", [])
    retry_count = state.get("retry_count", 0) + 1
    log.append(f"Retrying with new query (attempt {retry_count})...")
    return {**state, "retry_count": retry_count, "trace_log": log}

def give_up_node(state: RAGState) -> RAGState:
    log = state.get("trace_log", [])
    log.append("Max retries reached. Could not find a grounded answer.")
    final = "I don't have enough information in the provided documents to answer this question accurately."
    return {**state, "final_answer": final, "trace_log": log}

def build_graph() -> StateGraph:
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("critic", critic_node)
    graph.add_node("accept", accept_node)
    graph.add_node("retry", retry_node)
    graph.add_node("give_up", give_up_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "critic")

    graph.add_conditional_edges(
        "critic",
        decision_node,
        {
            "accept": "accept",
            "retry": "retry",
            "give_up": "give_up"
        }
    )

    graph.add_edge("retry", "retrieve")
    graph.add_edge("accept", END)
    graph.add_edge("give_up", END)

    return graph.compile()