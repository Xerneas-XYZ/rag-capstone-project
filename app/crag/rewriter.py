import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.config import get_settings

settings = get_settings()
llm = ChatOpenAI(model=settings.openai_model, temperature=0.3)


# 1. Update the prompt to output a structured JSON matching your maps
REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an advanced RAG query optimizer and metadata classifier for HomeShield Insurance documents.
Analyze the user's failed query and output a strict JSON object containing an optimized search query and metadata filters.

Your JSON structure MUST look exactly like this: {{
    "optimized_query": "string",
    "doc_type": "string or None",
    "policy_tier": "string or None"
}}

CRITICAL CLASSIFICATION RULES:
1. "policy_tier": Identify if the user mentions or implies a specific tier. Map to exactly one of these strings:
   - "standard" (if they mention basic cover, standard plan, or add-ons like accidental damage)
   - "comprehensive" (if they mention comprehensive plan or unlimited benefits)
   - "landlord" (if they mention tenants, rent recovery, or landlord plus)
   - If no tier is implied, return None.

2. "doc_type": Identify the source document category. Map to exactly one of these strings based on intent:
   - "claims_procedure" (if asking about: steps to take, evidence, what to do first, timelines, receipts, photos, or police reports)
   - "coverage_guide" (if asking about: high-value items, single article limits, valuables, or jewelry)
   - "home_policy_terms" (if asking about deep legal definitions, full terms, or general rules)
   - "home_insurance_glossary" (if asking for a definition of a specific word or insurance jargon)
   - "reference_guide" (if asking about specific exclusions, what is *not* covered, or peril definitions like storm windspeeds)
   - "standard" / "comprehensive" / "landlord" (if asking generally about basic table limits, premiums, or cover items for that tier)

3. "optimized_query": Rewrite the query using insurance keywords (e.g., translate "pipe leak" to "Escape of water", "break-in" to "Theft or attempted theft", "hotel" to "Alternative accommodation").

Return ONLY the JSON string. Do not include markdown code blocks, preambles, or explanations.""",
    ),
    (
        "human",
        "ORIGINAL QUERY: {query}\n"
        "ATTEMPT: {attempt_num}\n"
        "FAILURE REASON: {failure_reason}\n\n"
        "JSON Output:",
    ),
])




rewriter_chain = REWRITE_PROMPT | llm


def rewrite_query(query: str, attempt: int, failure_reason: str) -> str:
    """
    Rewrite a query that failed retrieval grading.

    Args:
        query:          Original user query.
        attempt:        Reflection iteration number (1 or 2).
        failure_reason: Short description of why grading failed.

    Returns:
        Rewritten query string.
    """
    result = rewriter_chain.invoke({
        "query": query,
        "attempt_num": attempt,
        "failure_reason": failure_reason,
    })
    return result.content.strip()








