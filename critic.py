from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

def get_client():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))

def critique_answer(question: str, answer: str, context_chunks: list) -> dict:
    context_text = "\n\n".join(context_chunks)

    prompt = f"""You are a strict answer quality evaluator.

Check if the ANSWER is fully grounded in the CONTEXT (no hallucination), actually answers the QUESTION, and is not vague or made-up.

QUESTION: {question}

CONTEXT:
{context_text}

ANSWER: {answer}

Respond ONLY in this exact format:
VERDICT: PASS or FAIL
REASON: one sentence explaining why
REFORMULATED_QUERY: if FAIL, write a better search query. If PASS, write NONE"""

    client = get_client()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )

    raw = response.choices[0].message.content.strip()
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