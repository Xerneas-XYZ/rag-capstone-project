from pydantic import BaseModel, Field
from typing import Optional, List


class BusinessContext(BaseModel):
    """Optional context about the user's business/coverage."""
    policy_tier: Optional[str] = Field(
        None, description="Policy tier: 'standard', 'comprehensive', 'landlord', etc."
    )
    industry: Optional[str] = Field(None, description="Industry type for contextual filtering")


class ChatRequest(BaseModel):
    """Request body for the /chat endpoint."""
    query: str = Field(..., description="The user's insurance question")
    session_id: Optional[str] = Field(
        None, description="Session ID for conversation tracking"
    )
    business_context: Optional[BusinessContext] = Field(
        None, description="Optional business/policy context"
    )


class CitationItem(BaseModel):
    """Citation/source metadata for an answer."""
    file: Optional[str] = Field(None, description="Source file name")
    section: Optional[str] = Field(None, description="Section title")
    page: Optional[int] = Field(None, description="Page number")


class ChatResponse(BaseModel):
    """Response body from the /chat endpoint."""
    answer: str = Field(..., description="The generated answer")
    confidence: str = Field(
        ..., description="Confidence level: HIGH, MEDIUM, or LOW"
    )
    sources: List[CitationItem] = Field(default_factory=list, description="Retrieved sources")
    reflections_used: int = Field(
        0, description="Number of query rewrites used"
    )
    used_web_search: bool = Field(
        False, description="Whether web search fallback was used"
    )
    compliance_passed: bool = Field(
        True, description="Whether compliance checks passed"
    )
    session_id: Optional[str] = Field(
        None, description="Session ID for conversation tracking"
    )
    pii_detected: bool = Field(
        False, description="Whether PII was detected and redacted"
    )
    mem0: Optional[str] = Field(
        None,
        description="Short-term session memory summary for the current conversation",
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Status: 'healthy' or 'degraded'")
    model: str = Field(..., description="OpenAI model in use")
    faiss_path: str = Field(..., description="Path to FAISS index")
