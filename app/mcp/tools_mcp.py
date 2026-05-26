import logging
import pandas as pd
from pathlib import Path
from functools import lru_cache
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from app.config import get_settings

settings = get_settings()

# ==========================================
# 0. CONFIGURE LOGGER
# ==========================================
logger = logging.getLogger("app.tools")

# ==========================================
# 1. HIGH-EFFICIENCY CSV CACHING
# ==========================================
@lru_cache(maxsize=3)
def load_csv_database(filename: str) -> pd.DataFrame:
    """Reads structured file records from disk into RAM."""
    logger.info(f"💾 CSV CACHE: Loading '{filename}' from disk into RAM memory.")
    file_path = Path(settings.csv_data_path) / filename
    return pd.read_csv(file_path)

def get_faiss_index():
    embeddings = OpenAIEmbeddings(model=settings.embedding_model, openai_api_key=settings.openai_api_key)
    return FAISS.load_local(settings.faiss_index_path, embeddings, allow_dangerous_deserialization=True)

# ==========================================
# 2. THE CONTRACTOR NETWORK LOOKUP TOOL
# ==========================================
@tool
def contractor_network_lookup(postcode: str, trade_type: str) -> str:
    """Finds approved contractors by postcode and trade type. Use this when the user asks for repairmen, builders, plumbers, etc."""
    logger.info(f"🔧 TOOL START: contractor_network_lookup called with postcode='{postcode}', trade_type='{trade_type}'")
    try:
        df = load_csv_database("ApprovedContractor_Network.csv")
        
        area_code = postcode.split()[0].upper()
        logger.info(f"🔍 CONTRACTOR FILTER: Stripped area code to '{area_code}' from full postcode '{postcode}'")
        
        mask = df['trade_type'].str.lower().str.contains(trade_type.lower(), na=False)
        mask &= df['coverage_postcodes'].str.contains(area_code, na=False, case=False)
        mask &= df['active_on_network'].str.lower() == 'yes'
        
        results = df[mask].head(3)
        logger.info(f"🎯 CONTRACTOR FILTER: Query finished. Found {len(results)} active network contractor rows matching criteria.")
        
        if not results.empty:
            response = f"Here are the approved {trade_type} contractors covering {postcode}:\n"
            for _, row in results.iterrows():
                response += f"- **{row['company_name']}** | 📞 {row['phone']} | ⭐ {row['rating_out_of_5']}/5 Rating | {row['emergency_callout']} \n"
            return response
        
        return f"No approved {trade_type} contractors found covering the {area_code} area at this time."
    except Exception as e:
        logger.error(f"❌ TOOL ERROR: contractor_network_lookup failed. Trace: {str(e)}", exc_info=True)
        return f"Contractor database is currently unavailable. Error: {str(e)}"

# ==========================================
# 3. THE DAMAGE COST ESTIMATOR
# ==========================================
@tool
def damage_cost_estimator(damage_type: str, property_size: str) -> str:
    """Returns indicative repair cost range based on damage type and property size."""
    logger.info(f"🔧 TOOL START: damage_cost_estimator called with damage_type='{damage_type}', property_size='{property_size}'")
    try:
        df = load_csv_database("PropertyDamage_RepairCostTable.csv")
        
        mask = df['damage_type'].str.lower().str.contains(damage_type.lower(), na=False)
        mask &= df['property_size_category'].str.lower().str.contains(property_size.lower(), na=False)
        
        result = df[mask].head(1)
        logger.info(f"🎯 COST FILTER: Query finished. Row found matching criteria: {not result.empty}")
        
        if not result.empty:
            row = result.iloc[0]
            low_cost = row['repair_cost_low_gbp']
            high_cost = row['repair_cost_high_gbp']
            avg_cost = row['repair_cost_avg_gbp']
            labour_days = row['typical_labour_days']
            
            return (
                f"**[INDICATIVE ESTIMATE ONLY - NOT A BINDING QUOTE]**\n"
                f"For {row['damage_type']} in a {row['property_size_category']}, the estimated repair cost is typically between "
                f"**£{low_cost} and £{high_cost}** (Average: £{avg_cost}).\n"
                f"This usually takes about {labour_days} days of labour. Note: {row['vat_included']}."
            )
        
        return "Could not find reliable cost data for this specific damage and property size."
    except Exception as e:
        logger.error(f"❌ TOOL ERROR: damage_cost_estimator failed. Trace: {str(e)}", exc_info=True)
        return f"Cost estimator database unavailable. Error: {str(e)}"

# ==========================================
# 4. UNDERINSURANCE / REBUILDING COST CHECKER
# ==========================================
@tool
def rebuilding_cost_estimator(property_type: str, region: str) -> str:
    """Checks the current rebuilding cost index benchmarks by property type and region to flag underinsurance."""
    logger.info(f"🔧 TOOL START: rebuilding_cost_estimator called with property_type='{property_type}', region='{region}'")
    try:
        df = load_csv_database("RebuildingCost_Index_ByPropertyType.csv")
        
        mask = df['property_type'].str.lower().str.contains(property_type.lower(), na=False)
        mask &= df['region'].str.lower().str.contains(region.lower(), na=False)
        
        result = df[mask].head(1)
        logger.info(f"🎯 REBUILD FILTER: Query finished. Row found matching criteria: {not result.empty}")
        
        if not result.empty:
            row = result.iloc[0]
            return (
                f"The benchmark rebuilding cost for a {row['property_type']} in {row['region']} is estimated between "
                f"**£{row['estimated_total_rebuild_low_gbp']} and £{row['estimated_total_rebuild_high_gbp']}**.\n"
                f"This includes {row['includes_professional_fees_pct']} and {row['includes_vat_pct']}."
            )
        return "Rebuilding benchmark not found for this property type and region."
    except Exception as e:
        logger.error(f"❌ TOOL ERROR: rebuilding_cost_estimator failed. Trace: {str(e)}", exc_info=True)
        return "Rebuilding cost index unavailable."

# ==========================================
# 5. CLAIM STATUS TRACKER (Mock DB)
# ==========================================
@tool
def claim_status_tracker(claim_id: str) -> str:
    """Returns the current status of a claim. Use this when the user asks about an existing claim."""
    logger.info(f"🔧 TOOL START: claim_status_tracker called with claim_id='{claim_id}'")
    
    formatted_id = claim_id.strip().upper()
    logger.info(f"🔍 CLAIM FILTER: Normalized tracking query look-up token to '{formatted_id}'")
    
    mock_claims_db = {
        "CLM-12345": {
            "status": "Under Review", 
            "pending_action": "Awaiting structural damage report from surveyor.",
            "next_steps": "Please upload the surveyor report to your portal."
        },
        "CLM-99823": {
            "status": "Approved", 
            "pending_action": "None",
            "next_steps": "Payment of £2,450 is currently processing and will reach your account in 3-5 days."
        },
        "CLM-44556": {
            "status": "Awaiting Excess Payment",
            "pending_action": "Policy excess of £250 is unpaid.",
            "next_steps": "Log in to the billing portal to clear the excess to allow contractors to proceed."
        }
    }
    
    claim = mock_claims_db.get(formatted_id)
    logger.info(f"🎯 CLAIM STATUS: Database extraction match found: {claim is not None}")
    
    if claim:
        return (
            f"**Claim Status for {formatted_id}:** {claim['status']}\n"
            f"- **Pending Actions:** {claim['pending_action']}\n"
            f"- **Next Steps:** {claim['next_steps']}"
        )
    
    return f"We could not locate Claim ID '{formatted_id}' in our system. Please verify the number and try again."

# ==========================================
# 6. EXPORT ALL TOOLS
# ==========================================
def get_tools():
    configured_tools = [
        contractor_network_lookup, 
        damage_cost_estimator, 
        rebuilding_cost_estimator, 
        claim_status_tracker
    ]
    return configured_tools