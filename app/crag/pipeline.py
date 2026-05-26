import logging
from dataclasses import dataclass, field
from app.crag.retriever import retrieve_chunks
from app.crag.grader import grade_all_chunks
from app.crag.rewriter import rewrite_query
from app.crag.generator import generate_response
from app.compliance.checker import compliance_check
from app.config import get_settings
from app.mcp.tools_mcp import get_tools  # Import your tools loader
from langchain_openai import ChatOpenAI
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

# Initialize an internal routing LLM bound with your tools
routing_llm = ChatOpenAI(model=settings.openai_model, temperature=0).bind_tools(get_tools())
TOOL_MAPPING = {tool.name: tool for tool in get_tools()}

async def run_crag(
    query: str,
    doc_type: str = None,
    policy_tier: str = None,
) -> CRAGResult:
    """
    Execute the Multi-Tool Routed Self-Reflective (Corrective) RAG pipeline.
    """
    current_query = query
    reflections = 0

    # ── STEP 0: UPFRONT INTENT ROUTING LAYER ──────────────────────────────────
    # Check if the query requires a structured database lookup before hitting document text
    logger.info(f"Checking upfront tool routing intent for query: '{query}'")
    route_check = routing_llm.invoke(f"Determine if this query requires a specialized tool. Query: {query}")
    
    if route_check.tool_calls:
        tool_call = route_check.tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        logger.info(f"⚡ INTENT ROUTER: Matching structural intent found. Running tool: '{tool_name}' with args {tool_args}")
        selected_tool = TOOL_MAPPING.get(tool_name)
        
        if selected_tool:
            try:
                tool_output = selected_tool.invoke(tool_args)
                # Compliance check the tool answer before finalizing output
                checked_answer, compliant = compliance_check(tool_output)
                
                return CRAGResult(
                    answer=checked_answer,
                    confidence="HIGH",
                    sources=[{"file": "Structured Database CSV", "section": tool_name, "page": 1}],
                    reflections=0,
                    used_web_search=False,
                    compliance_passed=compliant
                )
            except Exception as e:
                logger.error(f"Upfront tool execution failed: {e}")
                # Fallthrough to standard RAG if the database tool errors out

    # ── Step 1: Standard CRAG Text Document Loop ─────────────────────────────
    # (Runs only if no structural database tool was selected above)
    for attempt in range(settings.crag_max_reflections + 1):
        chunks = retrieve_chunks(current_query, doc_type, policy_tier, k=6)
        logger.info(f"Attempt {attempt + 1}: retrieved {len(chunks)} chunks")

        good_chunks, avg_score = grade_all_chunks(current_query, chunks)
        logger.info(
            f"Grading: {len(good_chunks)}/{len(chunks)} passed, "
            f"avg_combined={avg_score:.2f}, threshold={settings.crag_relevance_threshold}"
        )

        if good_chunks:
            break  # Sufficient quality — proceed to generate

        # ── Step 3b: Rewrite ─────────────────────────────────────────────
        # Inside your loop in app/crag/pipeline.py:
        if attempt < settings.crag_max_reflections:
            reflections += 1
            failure_reason = f"Only {len(good_chunks)} chunks passed threshold."
            
            # Call the updated rewriter
            rewrite_result = rewrite_query(query, attempt + 1, failure_reason)
            
            # 🚨 CHECK SAFETY NET: Did the rewriter discover that we need a tool instead?
            # 🚨 DIRECT INTERCEPTION: Check if a tool redirection is requested
            if rewrite_result.get("suggested_tool"):
                target_tool = rewrite_result["suggested_tool"]
                
                # 1. Extract the pre-compiled arguments from the rewriter dictionary
                extracted_args = rewrite_result.get("tool_args", {})
                
                logger.info(f"⚡ PIPELINE REDIRECTION: Routing to tool '{target_tool}' with args: {extracted_args}")
                
                selected_tool = TOOL_MAPPING.get(target_tool)
                
                if selected_tool:
                    try:
                        # 2. Pass the extracted arguments directly into the tool
                        tool_output = selected_tool.invoke(extracted_args)
                        
                        # 3. Process the output and return immediately
                        checked_answer, compliant = compliance_check(tool_output)
                        return CRAGResult(
                            answer=checked_answer,
                            confidence="HIGH",
                            sources=[{"file": "Structured Database CSV", "section": target_tool, "page": 1}],
                            reflections=reflections,
                            compliance_passed=compliant
                        )
                    except Exception as tool_exc:
                        logger.error(f"Redirected tool execution faulted: {tool_exc}")
            # Otherwise, update filters and continue to next text retrieval attempt
            current_query = rewrite_result.get("optimized_query", current_query)
            doc_type = rewrite_result.get("doc_type")
            policy_tier = rewrite_result.get("policy_tier")

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

    # ── Step 3a: Generate Policy Document Response ───────────────────────────
    confidence = "HIGH" if reflections == 0 else "MEDIUM"
    answer = generate_response(current_query, good_chunks, confidence)

    # ── Step 4: Compliance check ─────────────────────────────────────────
    checked_answer, compliant = compliance_check(answer)

    sources = [
        {
            "file":     c.metadata.get("source_file"),
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