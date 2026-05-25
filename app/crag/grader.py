from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from app.config import get_settings

settings = get_settings()
llm = ChatOpenAI(model=settings.openai_model, temperature=0)


class ChunkScore(BaseModel):
    relevance: float = Field(
        ..., ge=0.0, le=1.0,
        description="0–1: does this chunk address the query topic?",
    )
    specificity: float = Field(
        ..., ge=0.0, le=1.0,
        description="0–1: does this chunk contain the specific clause/limit/condition needed?",
    )
    reasoning: str = Field(
        ..., description="One-sentence explanation of the scores."
    )


GRADE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a commercial insurance document grader.
Score this retrieved text chunk for its usefulness in answering the query.
Return JSON with:
  relevance   (0–1): topical relevance to the query
  specificity (0–1): whether the chunk contains the specific clause, sub-limit, or condition needed
  reasoning   : one sentence explaining your scores""",
    ),
    ("human", "QUERY: {query}\n\nCHUNK:\n{chunk_text}"),
])

grader_chain = GRADE_PROMPT | llm.with_structured_output(ChunkScore)


def grade_chunk(query: str, chunk_text: str) -> ChunkScore:
    return grader_chain.invoke({"query": query, "chunk_text": chunk_text})


def grade_all_chunks(
    query: str,
    chunks: list,
    threshold: float = None,
) -> tuple[list, float]:
    """
    Grade all chunks and return those meeting the combined relevance threshold.

    Returns:
        (passing_chunks, average_combined_score)
    """
    t = threshold if threshold is not None else settings.crag_relevance_threshold
    passing = []
    total_score = 0.0

    for chunk in chunks:
        score = grade_chunk(query, chunk.page_content)
        chunk.metadata["grade"] = score.model_dump()
        combined = (score.relevance + score.specificity) / 2
        total_score += combined
        if combined >= t:
            passing.append(chunk)

    avg = total_score / len(chunks) if chunks else 0.0
    return passing, avg
