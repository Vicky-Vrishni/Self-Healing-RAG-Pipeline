from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

def get_critic_llm():
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0.0
    )

def critique_answer(question: str, answer: str, context_chunks: list) -> dict:
    context_text = "\n\n".join([doc.page_content for doc in context_chunks])

    prompt = PromptTemplate(
        input_variables=["question", "answer", "context"],
        template="""
You are a strict answer quality evaluator.

Your job is to check if the given ANSWER is:
1. Fully grounded in the CONTEXT (no hallucination)
2. Actually answers the QUESTION asked
3. Not vague, incomplete, or made-up

QUESTION: {question}

CONTEXT:
{context}

ANSWER: {answer}

Respond ONLY in this exact format:
VERDICT: PASS or FAIL
REASON: one sentence explaining why
REFORMULATED_QUERY: if FAIL, write a better search query to find the answer. If PASS, write NONE
"""
    )

    llm = get_critic_llm()
    chain = prompt | llm
    response = chain.invoke({
        "question": question,
        "answer": answer,
        "context": context_text
    })

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
        "reformulated_query": reformulated_query
    }