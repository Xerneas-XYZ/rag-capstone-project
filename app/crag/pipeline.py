import logging
from dataclasses import dataclass, field
from app.crag.retriever import retrieve_chunks
from app.crag.grader import grade_all_chunks
from app.crag.rewriter import rewrite_query
from app.crag.generator import generate_response
from app.compliance.checker import compliance_check
from app.config import get_settings
from langchain_community.tools import DuckDuckGoSearchRun

settings = get_settings()
logger = logging.getLogger(__name__)
web_search = DuckDuckGoSearchRun()

DISCLAIMER = (
    "This information is for guidance only. "
    "Please consult your broker or insurer for binding coverage confirmation."
)


@dataclass
class CRAGResult:
    answer: str
    confidence: str          # HIGH | MEDIUM | LOW
    sources: list = field(default_factory=list)
    reflections: int = 0
    used_web_search: bool = False
    compliance_passed: bool = True


async def run_crag(
    query: str,
    doc_type: str = None,
    policy_tier: str = None,
) -> CRAGResult:
    """
    Execute the Self-Reflective (Corrective) RAG pipeline.

    Flow:
        1. Retrieve top-k chunks from FAISS
        2. Grade chunks — pass if combined score >= threshold
        3a. If pass: generate response
        3b. If fail: rewrite query and loop (max crag_max_reflections times)
        4. If all reflections exhausted: web search fallback
        5. Compliance check on generated answer
    """
    current_query = query
    reflections = 0

    for attempt in range(settings.crag_max_reflections + 1):
        # ── Step 1: Retrieve ──────────────────────────────────────────────
        chunks = retrieve_chunks(current_query, doc_type, policy_tier, k=6)
        logger.info(f"Attempt {attempt + 1}: retrieved {len(chunks)} chunks")

        # ── Step 2: Grade ────────────────────────────────────────────────
        good_chunks, avg_score = grade_all_chunks(current_query, chunks)
        logger.info(
            f"Grading: {len(good_chunks)}/{len(chunks)} passed, "
            f"avg_combined={avg_score:.2f}, threshold={settings.crag_relevance_threshold}"
        )

        if good_chunks:
            break  # Sufficient quality — proceed to generate

        # ── Step 3b: Rewrite ─────────────────────────────────────────────
        if attempt < settings.crag_max_reflections:
            reflections += 1
            failure_reason = (
                f"Only {len(good_chunks)}/{len(chunks)} chunks passed "
                f"relevance threshold {settings.crag_relevance_threshold}. "
                f"Average combined score: {avg_score:.2f}."
            )
            current_query = rewrite_query(query, attempt + 1, failure_reason)
            logger.info(f"Rewritten query (attempt {attempt + 1}): {current_query}")

        else:
            # ── Web search fallback ───────────────────────────────────────
            logger.warning("CRAG: all reflection attempts exhausted — triggering web search")
            try:
                web_result = web_search.run(query)
            except Exception as e:
                logger.error(f"Web search failed: {e}")
                web_result = "Web search unavailable."

            answer = (
                f"**[LOW CONFIDENCE — based on web search, not verified policy document]**\n\n"
                f"{web_result}\n\n"
                f"{DISCLAIMER}"
            )
            return CRAGResult(
                answer=answer,
                confidence="LOW",
                sources=[],
                reflections=reflections,
                used_web_search=True,
                compliance_passed=True,
            )

    # ── Step 3a: Generate ────────────────────────────────────────────────
    confidence = "HIGH" if reflections == 0 else "MEDIUM"
    answer = generate_response(current_query, good_chunks, confidence)

    # ── Step 4: Compliance check ─────────────────────────────────────────
    checked_answer, compliant = compliance_check(answer)

    sources = [
        {
            "file":    c.metadata.get("source_file"),
            "section": c.metadata.get("section_title"),
            "page":    c.metadata.get("page_num"),
        }
        for c in good_chunks
    ]

    return CRAGResult(
        answer=checked_answer,
        confidence=confidence,
        sources=sources,
        reflections=reflections,
        used_web_search=False,
        compliance_passed=compliant,
    )
