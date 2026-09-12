from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()


def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.0,
        api_key=os.getenv("GROQ_API_KEY"),
    )


CRITIC_PROMPT = ChatPromptTemplate.from_template(
    """You are a strict answer quality evaluator.

Check if the ANSWER is fully grounded in the CONTEXT (no hallucination), actually answers the QUESTION, and is not vague or made-up.

QUESTION: {question}

CONTEXT:
{context}

ANSWER: {answer}

Respond ONLY in this exact format:
VERDICT: PASS or FAIL
REASON: one sentence explaining why
REFORMULATED_QUERY: if FAIL, write a better search query. If PASS, write NONE"""
)


def critique_answer(question: str, answer: str, context_chunks: list) -> dict:
    context_text = "\n\n".join(context_chunks)

    llm = get_llm()
    chain = CRITIC_PROMPT | llm
    response = chain.invoke(
        {"question": question, "context": context_text, "answer": answer}
    )

    raw = response.content.strip()
    lines = raw.split("\n")

    verdict = "FAIL"
    reason = "Could not evaluate"
    reformulated_query = question

    for line in lines:
        if line.startswith("VERDICT:"):
            verdict = line.replace("VERDICT:", "").strip()
        elif line.startswith("REASON:"):
            reason = line.replace("REASON:", "").strip()
        elif line.startswith("REFORMULATED_QUERY:"):
            reformulated_query = line.replace("REFORMULATED_QUERY:", "").strip()

    if reformulated_query == "NONE":
        reformulated_query = question

    return {
        "verdict": verdict,
        "reason": reason,
        "reformulated_query": reformulated_query,
    }