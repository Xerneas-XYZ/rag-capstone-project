"""
ProShield Commercial Insurance AI Copilot — Streamlit Frontend
Run: streamlit run ui/app.py --server.port 8501
"""
import os
import uuid
import httpx
import streamlit as st
import logging
from datetime import datetime

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_URL = os.getenv("API_URL", "http://localhost:8000")
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "60"))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HomeShield Insurance AI Copilot",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Health check on startup ─────────────────────────────────────────────────────
@st.cache_resource
def check_api_health():
    """Check if the API is healthy on startup."""
    try:
        resp = httpx.get(f"{API_URL}/health", timeout=5)
        resp.raise_for_status()
        return True, None
    except Exception as e:
        return False, str(e)

# ── Custom CSS for clean white styling ───────────────────────────────────────
st.markdown("""
<style>
    .confidence-high   { color: #1A6B35; font-weight: bold; }
    .confidence-medium { color: #7A4E00; font-weight: bold; }
    .confidence-low    { color: #8B0000; font-weight: bold; }
    .disclaimer-box {
        background: #FFF3CD;
        border-left: 4px solid #7A4E00;
        padding: 8px 12px;
        font-size: 0.85em;
        border-radius: 4px;
        margin-top: 8px;
    }
    .source-box {
        background: #F0F4F8;
        border-left: 3px solid #0A5C5C;
        padding: 6px 10px;
        font-size: 0.82em;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## � HomeShield AI Copilot")
    st.markdown("Home Insurance AI Assistant")
    st.markdown("---")

    st.markdown("### Policy Preference")
    st.caption("Optional — helps filter policy responses to your profile.")

    policy_tier = st.selectbox("Policy Tier", ["", "standard", "comprehensive", "landlord"])

    st.markdown("---")
    st.markdown("### Confidence Legend")
    st.markdown("🟢 **HIGH** — fully sourced from policy documents")
    st.markdown("🟡 **MEDIUM** — partially sourced (query refinement used)")
    st.markdown("🔴 **LOW** — web search fallback; verify with your broker")

    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.caption("Session ID: " + st.session_state.get("session_id", "—")[:12] + "...")

# ── Session state ─────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("� HomeShield Insurance AI Copilot")
st.caption(
    "Powered by Self-Reflective RAG (CRAG) · "
    "Your AI assistant for home insurance coverage, claims, and policy questions"
)

# ── API Status Check ─────────────────────────────────────────────────────────
api_healthy, health_error = check_api_health()
if not api_healthy:
    st.warning(
        f"⚠️ **API Status:** Cannot connect to {API_URL} — {health_error}. "
        f"Make sure the FastAPI server is running (`uvicorn app.main:app --reload`)."
    )
else:
    st.success(f"✅ API is healthy ({API_URL})", icon="✅")

# ── Suggested queries ─────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("#### 💡 Try asking:")
    cols = st.columns(2)
    suggestions = [
        "What coverage does HomeShield provide for water damage?",
        "Are accidental damages covered under the standard plan?",
        "What's the claims process for a theft claim?",
        "What is the single article limit for jewellery?",
    ]
    for i, suggestion in enumerate(suggestions):
        with cols[i % 2]:
            if st.button(suggestion, key=f"sug_{i}"):
                st.session_state._pending_query = suggestion
                st.rerun()

# ── Display message history ───────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("meta"):
            m = msg["meta"]
            conf_map  = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}
            conf_icon = conf_map.get(m.get("confidence", ""), "⚪")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Confidence", m.get("confidence", "—"), conf_icon)
            col2.metric("Reflections", m.get("reflections", 0), "🔄")
            col3.metric("Web Search", "Yes" if m.get("web") else "No", "🌐")
            col4.metric("Compliance", "✅" if m.get("compliant") else "⚠️", "")
            
            if m.get("pii"):
                st.warning("⚠️ **PII Detection:** Personal information was detected and redacted from your query.")
            if m.get("sources"):
                with st.expander("📄 Source Citations", expanded=False):
                    for i, src in enumerate(m["sources"], 1):
                        st.markdown(
                            f"**{i}.** `{src.get('file', '?')}` — "
                            f"*{src.get('section', '?')}* (p. {src.get('page', '?')})",
                        )

# ── Handle pending suggestion click ──────────────────────────────────────────
pending = st.session_state.pop("_pending_query", None)

# ── Chat input ────────────────────────────────────────────────────────────────
prompt = st.chat_input("Ask a commercial insurance question...") or pending

if prompt:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call API
    with st.chat_message("assistant"):
        with st.spinner("Analysing policy documents..."):
            # Build business context with only non-empty fields
            business_context = {}
            if policy_tier and policy_tier != "":
                business_context["policy_tier"] = policy_tier
            
            payload = {
                "query": prompt,
                "session_id": st.session_state.session_id,
                "business_context": business_context if business_context else None,
            }
            
            logger.info(f"Sending request: {payload}")
            
            try:
                resp = httpx.post(
                    f"{API_URL}/chat",
                    json=payload,
                    timeout=API_TIMEOUT,
                )
                resp.raise_for_status()
                data = resp.json()

                # Display the answer
                st.markdown(data["answer"])

                # Display metadata
                conf_map  = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}
                conf_icon = conf_map.get(data.get("confidence", ""), "⚪")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Confidence", data.get("confidence", "—"), conf_icon)
                col2.metric("Reflections", data.get("reflections_used", 0), "🔄")
                col3.metric("Web Search", "Yes" if data.get("used_web_search") else "No", "🌐")
                col4.metric("Compliance", "✅" if data.get("compliance_passed") else "⚠️", "")

                if data.get("pii_detected"):
                    st.warning("⚠️ **PII Detection:** Personal information was detected and redacted from your query.")

                if data.get("sources"):
                    with st.expander("📄 Source Citations", expanded=False):
                        for i, src in enumerate(data["sources"], 1):
                            st.markdown(
                                f"**{i}.** `{src.get('file', '?')}` — "
                                f"*{src.get('section', '?')}* (p. {src.get('page', '?')})",
                            )

                # Store message with metadata
                meta = {
                    "confidence":  data.get("confidence"),
                    "reflections": data.get("reflections_used", 0),
                    "web":         data.get("used_web_search", False),
                    "compliant":   data.get("compliance_passed", True),
                    "pii":         data.get("pii_detected", False),
                    "sources":     data.get("sources", []),
                }
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": data["answer"],
                    "meta": meta,
                })

            except httpx.ConnectError:
                err = (
                    "❌ **Connection Error** — Cannot connect to the API. "
                    f"Is FastAPI running on {API_URL}?"
                )
                st.error(err)
                logger.error(f"Connection error: {err}")
                st.session_state.messages.append({"role": "assistant", "content": err})
            
            except httpx.TimeoutException:
                err = (
                    f"⏱️ **Timeout Error** — The API took longer than {API_TIMEOUT}s to respond. "
                    "Please try again."
                )
                st.error(err)
                logger.error(f"Timeout error: {err}")
                st.session_state.messages.append({"role": "assistant", "content": err})
            
            except httpx.HTTPStatusError as e:
                detail = e.response.json().get("detail", str(e)) if e.response.text else str(e)
                err = f"❌ **API Error** ({e.response.status_code}): {detail}"
                st.error(err)
                logger.error(f"HTTP error: {err}")
                st.session_state.messages.append({"role": "assistant", "content": err})
            
            except Exception as e:
                err = f"❌ **Unexpected Error:** {str(e)}"
                st.error(err)
                logger.exception(f"Unexpected error: {err}")
                st.session_state.messages.append({"role": "assistant", "content": err})

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
footer_cols = st.columns([1, 1, 1, 1])
# footer_cols[0].caption("📧 **Support:** contact@proshield.ai")
# footer_cols[1].caption("📚 **Docs:** [Policy Coverage Guide](https://docs.proshield.ai)")
footer_cols[2].caption("⚖️ **Disclaimer:** Guidance only — consult your broker for binding confirmation")
footer_cols[3].caption("🔐 **Session ID:** " + st.session_state.get("session_id", "—")[:8])
