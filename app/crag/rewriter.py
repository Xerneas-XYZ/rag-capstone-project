import logging
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# ==========================================================
# 1. DEFINE EXPLICIT PARAMETER STRUCURES FOR OPENAI COMPLIANCE
# ==========================================================
class ToolArgs(BaseModel):
    """
    Explicitly defines all possible parameters across all tools.
    This guarantees compliance with OpenAI's strict schema rules.
    """
    postcode: Optional[str] = Field(default=None, description="E.g., 'SW6', 'M1'. Prefix area code.")
    trade_type: Optional[str] = Field(default=None, description="E.g., 'Roofing', 'Plumbing', 'Building'.")
    damage_type: Optional[str] = Field(default=None, description="E.g., 'Roof tile replacement'.")
    property_size: Optional[str] = Field(default=None, description="E.g., '3-bed semi-detached house'.")
    property_type: Optional[str] = Field(default=None, description="E.g., '1-bed flat', '3-bed semi'.")
    region: Optional[str] = Field(default=None, description="E.g., 'London', 'South East England'.")
    claim_id: Optional[str] = Field(default=None, description="E.g., 'CLM-12345'.")

class RewriteOutput(BaseModel):
    """Structured JSON schema for query optimization and metadata routing."""
    optimized_query: str = Field(
        description="The rewritten query using standard technical insurance vocabulary."
    )
    doc_type: Optional[str] = Field(
        default=None, 
        description="Mapped document category string: 'claims_procedure', 'coverage_guide', 'home_policy_terms', 'home_insurance_glossary', 'reference_guide', or None."
    )
    policy_tier: Optional[str] = Field(
        default=None, 
        description="Identified insurance tier string: 'standard', 'comprehensive', 'landlord', or None."
    )
    suggested_tool: Optional[str] = Field(
        default=None, 
        description="Target database tool: 'contractor_network_lookup', 'damage_cost_estimator', 'rebuilding_cost_estimator', 'claim_status_tracker', or None."
    )
    tool_args: Optional[ToolArgs] = Field(
        default=None,
        description="Structured key-value arguments container required by the target tool. Leave completely empty if suggested_tool is null."
    )

# ==========================================================
# 2. THE ROUTING PROMPT
# ==========================================================
REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an advanced RAG query optimizer, metadata classifier, and tool routing validator for HomeShield Insurance documents.
Analyze the user's failed query, identify their core operational intent, and determine if a database lookup tool is necessary instead of document text RAG.

=== STRUCTURAL TOOL MATCHING & PARAMETER SCHEMAS ===
If a tool is selected, you MUST populate "suggested_tool" with the tool name AND populate the corresponding parameters inside the "tool_args" object:

1. Tool Name: "contractor_network_lookup"
   - Use cases: User needs tradespeople, plumbers, builders, or local repair networks.
   - Populates inside tool_args: "postcode" and "trade_type"

2. Tool Name: "damage_cost_estimator"
   - Use cases: User needs repair cost estimates, pricing ranges, or valuation bounds for specific damage.
   - Populates inside tool_args: "damage_type" and "property_size"

3. Tool Name: "rebuilding_cost_estimator"
   - Use cases: User needs regional structural rebuilding index benchmarks or underinsurance risk checks.
   - Populates inside tool_args: "property_type" and "region"

4. Tool Name: "claim_status_tracker"
   - Use cases: User is providing or checking a specific claim ID token.
   - Populates inside tool_args: "claim_id"

=== CONVERSATIONAL DOCUMENT RAG CRITERIA ===
- If the query is genuinely a document question (coverage rules, policy exclusions, contract timelines, legal definitions), set "suggested_tool" and "tool_args" to null.
- Map "policy_tier" to: 'standard', 'comprehensive', or 'landlord'.
- Map "doc_type" to: 'claims_procedure', 'coverage_guide', 'home_policy_terms', 'home_insurance_glossary', or 'reference_guide'.""",
    ),
    (
        "human",
        "ORIGINAL QUERY: {query}\n"
        "ATTEMPT: {attempt_num}\n"
        "FAILURE REASON: {failure_reason}\n",
    ),
])

# Initialize the structured optimization chain
llm = ChatOpenAI(model=settings.openai_model, temperature=0.6)
rewriter_chain = REWRITE_PROMPT | llm.with_structured_output(RewriteOutput)

# ==========================================================
# 3. EXECUTABLE REWRITE ROUTINE
# ==========================================================
def rewrite_query(query: str, attempt: int, failure_reason: str) -> dict:
    """
    Optimizes a failed query, extracts structural metadata, and packages 
    the exact arguments required by the target tool to prevent validation faults.
    """
    logger.info(f"🔄 REWRITER: Optimizing query parameters for iteration attempt {attempt}...")
    try:
        structured_response = rewriter_chain.invoke({
            "query": query,
            "attempt_num": attempt,
            "failure_reason": failure_reason,
        })
        
        # Convert the structural model output into a base dictionary
        output_dict = {
            "optimized_query": structured_response.optimized_query,
            "doc_type": structured_response.doc_type,
            "policy_tier": structured_response.policy_tier,
            "suggested_tool": structured_response.suggested_tool,
            "tool_args": {}
        }

        assert output_dict
        
        # Unpack only the active parameters that were generated into a clean flat dictionary
        if structured_response.tool_args:
            output_dict["tool_args"] = {
                k: v for k, v in structured_response.tool_args.model_dump().items() if v is not None
            }
            
        logger.info(f"📋 REWRITER ANALYSIS: Generated complete context payload -> {output_dict}")
        return output_dict
        
    except Exception as e:
        logger.error(f"❌ REWRITER FAULT: Falling back to clean state. Trace: {e}", exc_info=True)
        return {
            "optimized_query": query,
            "doc_type": None,
            "policy_tier": None,
            "suggested_tool": None,
            "tool_args": {}
        }