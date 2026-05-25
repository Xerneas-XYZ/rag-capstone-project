import logging
import sys
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.documents import Document
from app.models import ChatRequest, ChatResponse, CitationItem, HealthResponse
from app.crag.pipeline import run_crag
from app.guardrails.rails import (
    scan_input_pii, redact_pii,
    check_prompt_injection, check_off_topic,
    apply_output_guardrails,
)
from app.memory import append_session_memory, get_mem0_summary
from app.config import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
settings = get_settings()

app = FastAPI(
    title="HomeShield Insurance AI Copilot",
    version="1.0",
    description=(
        "Self-Reflective RAG (CRAG) copilot for commercial and business insurance. "
        "Covers: Commercial Property, Public & Employers Liability, Professional Indemnity, "
        "Cyber Liability, Business Interruption, D&O Liability."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model=settings.openai_model,
        faiss_path=settings.faiss_index_path,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Main chat endpoint. Accepts a commercial insurance query and returns a
    CRAG-generated, compliance-checked, guardrail-screened response.
    """
    # ── 1. Input guardrails ───────────────────────────────────────────────
    if check_prompt_injection(req.query):
        return ChatResponse(
            answer="I cannot process that request. Please ask a commercial insurance question.",
            confidence="HIGH",
            sources=[],
            reflections_used=0,
            used_web_search=False,
            compliance_passed=True,
            session_id=req.session_id,
        )

    if check_off_topic(req.query):
        return ChatResponse(
            answer=(
                "I can only assist with commercial and business insurance questions. "
                "Please ask about coverage, claims, premiums, risk assessment, or compliance."
            ),
            confidence="HIGH",
            sources=[],
            reflections_used=0,
            used_web_search=False,
            compliance_passed=True,
            session_id=req.session_id,
        )

    has_pii, pii_types = scan_input_pii(req.query)
    clean_query = redact_pii(req.query) if has_pii else req.query
    if has_pii:
        logger.warning(f"PII detected and redacted: {pii_types}")

    session_id = req.session_id or str(uuid.uuid4())

    # ── 2. Extract business context for CRAG metadata filtering ──────────
    policy_tier = None
    if req.business_context and hasattr(req.business_context, 'policy_tier'):
        policy_tier = req.business_context.policy_tier

    # ── 3. Run CRAG pipeline ──────────────────────────────────────────────
    try:
        result = await run_crag(clean_query, doc_type=None, policy_tier=policy_tier)
    except Exception as e:
        logger.error(f"CRAG pipeline error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Pipeline error — please retry.")

    # ── 4. Output guardrails ──────────────────────────────────────────────
    # Convert result.sources dict to Document objects for source grounding check
    source_docs = []
    for source in result.sources:
        doc = Document(
            page_content="",
            metadata={
                "source_file": source.get("file"),
                "section_title": source.get("section"),
                "page_num": source.get("page"),
            },
        )
        source_docs.append(doc)
    safe_answer, guardrail_passed = apply_output_guardrails(result.answer, source_docs)

    append_session_memory(session_id, "user", clean_query)
    append_session_memory(session_id, "assistant", safe_answer)
    mem0_summary = get_mem0_summary(session_id)

    return ChatResponse(
        answer=safe_answer,
        confidence=result.confidence,
        sources=[CitationItem(**s) for s in result.sources],
        reflections_used=result.reflections,
        used_web_search=result.used_web_search,
        compliance_passed=result.compliance_passed and guardrail_passed,
        session_id=session_id,
        pii_detected=has_pii,
        mem0=mem0_summary,
    )


@app.post("/ingest")
def trigger_ingest():
    """
    Trigger a background re-index of the data/pdfs directory.
    
    Returns:
        Success message with job ID.
    """
    import subprocess
    import os
    from pathlib import Path
    
    try:
        # Get the absolute path to the ingest script
        script_path = Path(__file__).parent.parent / "ingest" / "build_index.py"
        
        if not script_path.exists():
            logger.error(f"Ingest script not found: {script_path}")
            raise HTTPException(status_code=404, detail="Ingest script not found.")
        
        # Start subprocess with explicit python interpreter
        process = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        logger.info(f"Ingestion process started (PID: {process.pid})")
        
        return {
            "message": "Ingestion started in background. Check logs for progress.",
            "process_id": process.pid,
        }
    except Exception as e:
        logger.error(f"Failed to start ingest process: {e}")
        raise HTTPException(status_code=500, detail="Failed to start ingestion.")
