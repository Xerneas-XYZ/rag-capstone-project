import importlib
import sys
import types

import pandas as pd


def import_tools_fresh():
    if "app.mcp.tools.tools" in sys.modules:
        del sys.modules["app.mcp.tools.tools"]
    return importlib.import_module("app.mcp.tools.tools")


def call_tool(tool_obj, *args, **kwargs):
    if callable(tool_obj):
        return tool_obj(*args, **kwargs)
    if hasattr(tool_obj, "func"):
        return tool_obj.func(*args, **kwargs)
    if hasattr(tool_obj, "run"):
        return tool_obj.run(*args, **kwargs)
    raise RuntimeError("Unable to call wrapped tool object")


def test_contractor_network_lookup_matches():
    tools = import_tools_fresh()

    df = pd.DataFrame([
        {
            "trade_type": "Plumber",
            "coverage_postcodes": "SW6,SW7",
            "active_on_network": "Yes",
            "company_name": "Fast Plumbers Ltd",
            "phone": "01234 567890",
            "rating_out_of_5": 4.7,
            "emergency_callout": "24/7"
        }
    ])

    tools.load_csv_database = lambda filename: df
    out = call_tool(tools.contractor_network_lookup, "SW6 2AA", "plumber")

    assert "Fast Plumbers Ltd" in out
    assert "01234 567890" in out
    assert "4.7" in out or "4.7/5" in out


def test_contractor_network_lookup_no_match():
    tools = import_tools_fresh()
    df = pd.DataFrame([], columns=["trade_type", "coverage_postcodes", "active_on_network"])
    tools.load_csv_database = lambda filename: df

    out = call_tool(tools.contractor_network_lookup, "SW1 1AA", "electrician")
    assert "No approved electrician" in out


def test_damage_cost_estimator_found():
    tools = import_tools_fresh()
    df = pd.DataFrame([
        {
            "damage_type": "escape of water",
            "property_size_category": "small",
            "repair_cost_low_gbp": 200,
            "repair_cost_high_gbp": 800,
            "repair_cost_avg_gbp": 500,
            "typical_labour_days": 2,
            "vat_included": "VAT included"
        }
    ])

    tools.load_csv_database = lambda filename: df
    out = call_tool(tools.damage_cost_estimator, "pipe leak", "small")

    assert "INDICATIVE ESTIMATE ONLY" in out
    assert "£200" in out
    assert "£800" in out


def test_damage_cost_estimator_not_found():
    tools = import_tools_fresh()
    df = pd.DataFrame([], columns=["damage_type", "property_size_category"])
    tools.load_csv_database = lambda filename: df

    out = call_tool(tools.damage_cost_estimator, "meteor strike", "huge")
    assert "Could not find reliable cost data" in out


def test_rebuilding_cost_estimator_found():
    tools = import_tools_fresh()
    df = pd.DataFrame([
        {
            "property_type": "Detached",
            "region": "London",
            "estimated_total_rebuild_low_gbp": 200000,
            "estimated_total_rebuild_high_gbp": 350000,
            "includes_professional_fees_pct": "incl. fees 10%",
            "includes_vat_pct": "incl. VAT 20%",
        }
    ])

    tools.load_csv_database = lambda filename: df
    out = call_tool(tools.rebuilding_cost_estimator, "detached", "london")

    assert "£200000" in out
    assert "£350000" in out


def test_rebuilding_cost_estimator_not_found():
    tools = import_tools_fresh()
    df = pd.DataFrame([], columns=["property_type", "region"])
    tools.load_csv_database = lambda filename: df

    out = call_tool(tools.rebuilding_cost_estimator, "flat", "unknown")
    assert "Rebuilding benchmark not found" in out


def test_claim_status_tracker_known_and_unknown():
    tools = import_tools_fresh()
    out = call_tool(tools.claim_status_tracker, "clm-12345")
    assert "Under Review" in out

    out2 = call_tool(tools.claim_status_tracker, "does-not-exist")
    assert "could not locate" in out2.lower()
