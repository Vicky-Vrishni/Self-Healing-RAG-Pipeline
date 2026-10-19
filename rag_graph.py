
import os

import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.exceptions import OutputParserException
from groq import APIConnectionError, APITimeoutError, RateLimitError, APIStatusError 

from retriever import retrieve_chunks
from critic import critique_answer

load_dotenv()

MAX_RETRIES = 3 

FALLBACK_ANSWER = (
    "I don't have enough information about provided documents "
    "to answer this question accurately."
)


# ==========================================================
# LLM Configuration
# ==========================================================

def get_llm():
    """Create the Groq-hosted LLM instance."""
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Please configure it."
        )

    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.2,
        api_key=api_key,
        max_retries=2,
        timeout=30,
    )


# ==========================================================
# Answer Generation
# ==========================================================

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Answer the question using ONLY
the provided context.

If the context does not contain enough information, say exactly:
"I don't have enough information in the provided documents to answer this question."

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
)


def generate_answer(question: str, context_chunks: list) -> str:
    """Generate an answer using retrieved context."""
    context_text = "\n\n".join(context_chunks)

    llm = get_llm()
    chain = ANSWER_PROMPT | llm

    response = chain.invoke(
        {
            "context": context_text,
            "question": question,
        }
    )

    content = response.content

    if not isinstance(content, str) or not content.strip():
        raise ValueError("The LLM returned an empty or invalid response.")

    return content.strip()


# ==========================================================
# Self-Healing RAG Pipeline
# ==========================================================

def run_self_healing_rag(question: str) -> dict:
    """Retrieve, generate, critique, and retry when necessary."""

    trace_log = []
    current_query = question
    retry_count = 0
    context_chunks = []

    if not isinstance(question, str) or not question.strip():
        return {
            "final_answer": "Please enter a valid, non-empty question.",
            "verdict": "FAIL",
            "retry_count": 0,
            "trace_log": ["Invalid or empty question received."],
            "context": [],
        }

    try:
        while retry_count <= MAX_RETRIES:

            # Step 1: Retrieve relevant context
            try:
                context_chunks = retrieve_chunks(current_query, k=5)

                if not context_chunks:
                    trace_log.append(
                        f"No relevant chunks retrieved for query: '{current_query}'"
                    )
                else:
                    trace_log.append(
                        f"Retrieved {len(context_chunks)} chunks "
                        f"for query: '{current_query}'"
                    )

            except Exception:
                trace_log.append("Document retrieval failed.")
                st.error(
                    "Could not retrieve document chunks. "
                    "Check your vector index and document setup."
                )

                return {
                    "final_answer": FALLBACK_ANSWER,
                    "verdict": "FAIL",
                    "retry_count": retry_count,
                    "trace_log": trace_log,
                    "context": [],
                }

            # Step 2: Generate answer
            try:
                answer = generate_answer(question, context_chunks)
                trace_log.append(
                    f"Generated answer (attempt {retry_count + 1})."
                )

            except (RateLimitError, APITimeoutError, APIConnectionError):
                trace_log.append("Groq API rate limit, timeout, or connection error.")
                st.error(
                    "The AI service is temporarily unavailable. "
                    "Please try again shortly."
                )

                return {
                    "final_answer": (
                        "The AI service is temporarily unavailable. "
                        "Please try again later."
                    ),
                    "verdict": "ERROR",
                    "retry_count": retry_count,
                    "trace_log": trace_log,
                    "context": context_chunks,
                }

            except APIStatusError as error:
                trace_log.append(
                    f"Groq API returned HTTP status {error.status_code}."
                )
                st.error(
                    "The AI service returned an API error. "
                    "Please check your API configuration or try again later."
                )

                return {
                    "final_answer": (
                        "An AI service error occurred. Please try again later."
                    ),
                    "verdict": "ERROR",
                    "retry_count": retry_count,
                    "trace_log": trace_log,
                    "context": context_chunks,
                }

            except (ValueError, OutputParserException):
                trace_log.append("Invalid or empty LLM response.")
                st.error("The model returned an invalid response.")

                return {
                    "final_answer": FALLBACK_ANSWER,
                    "verdict": "FAIL",
                    "retry_count": retry_count,
                    "trace_log": trace_log,
                    "context": context_chunks,
                }

            # Step 3: Critic validation
            try:
                result = critique_answer(
                    question,
                    answer,
                    context_chunks,
                )

                trace_log.append(
                    f"Critic verdict: {result['verdict']} — "
                    f"{result.get('reason', 'No reason provided.')}"
                )

            except Exception:
                trace_log.append("Critic evaluation failed.")
                st.error("The answer validation step failed.")

                return {
                    "final_answer": FALLBACK_ANSWER,
                    "verdict": "ERROR",
                    "retry_count": retry_count,
                    "trace_log": trace_log,
                    "context": context_chunks,
                }

            # Step 4: Accept grounded answer
            if result.get("verdict") == "PASS":
                trace_log.append("Answer accepted by critic.")

                return {
                    "final_answer": answer,
                    "verdict": "PASS",
                    "retry_count": retry_count,
                    "trace_log": trace_log,
                    "context": context_chunks,
                }

            # Step 5: Retry if critic rejects the answer
            if retry_count >= MAX_RETRIES:
                trace_log.append(
                    "Maximum retries reached. No grounded answer was accepted."
                )

                return {
                    "final_answer": FALLBACK_ANSWER,
                    "verdict": "FAIL",
                    "retry_count": retry_count,
                    "trace_log": trace_log,
                    "context": context_chunks,
                }

            reformulated_query = result.get("reformulated_query")

            if (
                not isinstance(reformulated_query, str)
                or not reformulated_query.strip()
                or reformulated_query.strip() == current_query.strip()
            ):
                trace_log.append(
                    "Critic did not provide a valid new query. Stopping retries."
                )

                return {
                    "final_answer": FALLBACK_ANSWER,
                    "verdict": "FAIL",
                    "retry_count": retry_count,
                    "trace_log": trace_log,
                    "context": context_chunks,
                }

            current_query = reformulated_query.strip()
            retry_count += 1

            trace_log.append(
                f"Retrying with reformulated query (retry {retry_count})."
            )

    except Exception:
        trace_log.append("Unexpected internal pipeline error.")
        st.error("An unexpected error occurred in the RAG pipeline.")

        return {
            "final_answer": FALLBACK_ANSWER,
            "verdict": "ERROR",
            "retry_count": retry_count,
            "trace_log": trace_log,
            "context": context_chunks,
        }

    return {
        "final_answer": FALLBACK_ANSWER,
        "verdict": "FAIL",
        "retry_count": retry_count,
        "trace_log": trace_log,
        "context": context_chunks,
    }
