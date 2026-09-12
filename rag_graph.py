from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from retriever import retrieve_chunks
from critic import critique_answer
from dotenv import load_dotenv
import os

load_dotenv()

MAX_RETRIES = 3


def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.2,
        api_key=os.getenv("GROQ_API_KEY"),
    )


ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Answer the question using ONLY the provided context.
If the context does not contain enough information, say exactly:
"I don't have enough information in the provided documents to answer this question."

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
)


def generate_answer(question: str, context_chunks: list) -> str:
    context_text = "\n\n".join(context_chunks)
    llm = get_llm()
    chain = ANSWER_PROMPT | llm
    response = chain.invoke({"context": context_text, "question": question})
    return response.content.strip()


def run_self_healing_rag(question: str) -> dict:
    trace_log = []
    current_query = question
    retry_count = 0

    while retry_count <= MAX_RETRIES:
        context_chunks = retrieve_chunks(current_query, k=5)
        trace_log.append(f"Retrieved {len(context_chunks)} chunks for query: '{current_query}'")

        answer = generate_answer(question, context_chunks)
        trace_log.append(f"Generated answer (attempt {retry_count + 1})")

        result = critique_answer(question, answer, context_chunks)
        trace_log.append(f"Critic verdict: {result['verdict']} — {result['reason']}")

        if result["verdict"] == "PASS":
            trace_log.append("Answer accepted by critic.")
            return {
                "final_answer": answer,
                "verdict": "PASS",
                "retry_count": retry_count,
                "trace_log": trace_log,
                "context": context_chunks,
            }

        if retry_count >= MAX_RETRIES:
            trace_log.append("Max retries reached. Could not find a grounded answer.")
            return {
                "final_answer": "I don't have enough information in the provided documents to answer this question accurately.",
                "verdict": "FAIL",
                "retry_count": retry_count,
                "trace_log": trace_log,
                "context": context_chunks,
            }

        current_query = result["reformulated_query"]
        retry_count += 1
        trace_log.append(f"Retrying with new query (attempt {retry_count})...")

    return {
        "final_answer": "Unexpected error in RAG pipeline.",
        "verdict": "FAIL",
        "retry_count": retry_count,
        "trace_log": trace_log,
        "context": [],
    }