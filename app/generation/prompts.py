"""
Prompt construction.

Kept separate from llm_client.py so prompt wording can be iterated on
(and unit-tested) without touching networking code.
"""

from __future__ import annotations

from app.memory.conversation import Turn
from app.retrieval.reranker import RerankedChunk

REFUSAL_TEXT = "I couldn't find this information in the provided 3GPP specifications."

SYSTEM_INSTRUCTIONS = """You are a 3GPP standards question-answering assistant.

Answer the user's question using ONLY the numbered CONTEXT passages below.

Rules:
1. Do not use outside knowledge, even if you believe you know the answer.
2. Do not invent facts, clause numbers, spec IDs, tables, values, ranges, or procedures.
3. Every factual sentence in your answer must be directly supported by at least one CONTEXT passage.
4. When you state a fact, mention which passage number ([1], [2], ...) it came from.
5. If the answer is not explicitly supported by the CONTEXT, respond with exactly this sentence \
and nothing else:
{refusal}
""".format(refusal=REFUSAL_TEXT)


def format_context(passages: list[RerankedChunk]) -> str:
    return "\n\n".join(f"[{i + 1}] {p.content}" for i, p in enumerate(passages))


def format_history(history: list[Turn], max_turns: int) -> str:
    if not history:
        return ""
    trimmed = history[-max_turns:] if max_turns else []
    lines = []
    for turn in trimmed:
        lines.append(f"User: {turn.user}")
        lines.append(f"Assistant: {turn.assistant}")
    return "\n".join(lines)


def build_prompt(question: str, passages: list[RerankedChunk], history: list[Turn], max_history_turns: int) -> str:
    context_block = format_context(passages)
    history_block = format_history(history, max_history_turns)

    history_section = f"\nCONVERSATION SO FAR:\n{history_block}\n" if history_block else ""

    return f"""{SYSTEM_INSTRUCTIONS}
CONTEXT:

{context_block}
{history_section}
USER QUESTION:

{question}

ANSWER:
"""
