import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.config import get_settings

settings = get_settings()
logger = logging.getLogger("app.crag.generator")

# Use a clean base LLM. The generator synthesizes text context; it doesn't need tools anymore!
llm = ChatOpenAI(model=settings.openai_model, temperature=0)

GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are the Pro Home Insurance AI Copilot.
Answer the user's insurance question using ONLY the retrieved context provided below.

STRICT RULES:
1. Cite every factual claim: [Source: <filename>, <section>, p.<page>]
2. If the answer is not in the context, say "I could not find this information in the policy documents" — do NOT guess or infer.
3. Label your confidence at the start: **Confidence: HIGH** (all claims sourced) | **MEDIUM** (partially sourced) | **LOW** (inferred or web).
4. ALWAYS end with exactly this disclaimer:
   "This information is for guidance only. Please consult your broker or insurer for binding coverage confirmation."
5. Never confirm coverage, guarantee premiums, or give legal advice.
6. Keep the response concise and structured — use bullet points for lists of conditions or exclusions.""",
    ),
    ("human", "RETRIEVED CONTEXT:\n{context}\n\nQUESTION: {query}"),
])

generate_chain = GENERATE_PROMPT | llm

def format_context(chunks: list) -> str:
    """Format retrieved chunks into a clearly labelled context block."""
    parts = []
    for chunk in chunks:
        m = chunk.metadata
        source_label = (
            f"[Source: {m.get('source_file', 'Unknown')}, "
            f"{m.get('section_title', 'Unknown section')}, "
            f"p.{m.get('page_num', '?')}]"
        )
        parts.append(f"{source_label}\n{chunk.page_content}")
    return "\n\n---\n\n".join(parts)

def generate_response(query: str, chunks: list, confidence: str = "HIGH") -> str:
    """Generate a grounded, cited response from retrieved chunks."""
    context = format_context(chunks)
    logger.info("📝 GENERATOR: Synthesizing standard verified policy response.")
    result = generate_chain.invoke({"context": context, "query": query})
    return result.content