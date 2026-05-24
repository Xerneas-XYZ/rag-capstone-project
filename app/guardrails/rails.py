import re
import logging

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "This information is for guidance only. "
    "Please consult your broker or insurer for binding coverage confirmation."
)

# ── PII patterns ──────────────────────────────────────────────────────────────
PII_PATTERNS = {
    "UK Phone Number":               r"(?:07|\+44)\d{9,10}",
    "Indian Phone Number":           r"(?:\+91|0)?[6789]\d{9}",
    "Email Address":                 r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    "Bank Account Number (8-digit)": r"\b\d{8}\b",
    "Sort Code":                     r"\b\d{2}-\d{2}-\d{2}\b",
}

# ── Output danger patterns ────────────────────────────────────────────────────
BINDING_COVERAGE_PATTERNS = [
    r"you\s+are\s+(?:fully\s+)?covered",
    r"(?:your|this)\s+policy\s+(?:will\s+)?covers?",
    r"I\s+confirm\s+(?:your\s+)?(?:claim|coverage)",
    r"your\s+policy\s+will\s+pay",
    r"(?:your|this)\s+claim\s+is\s+approved",
]

GUARANTEED_PREMIUM_PATTERNS = [
    r"your\s+(?:exact\s+|final\s+|annual\s+)?premium\s+(?:is|will\s+be)",
    r"you\s+will\s+(?:pay|owe)\s+(?:exactly|approximately)?\s*[£$€]\d",
    r"the\s+(?:cost|price|premium)\s+(?:is|will\s+be)\s+[£$€]\d",
]

LEGAL_ADVICE_PATTERNS = [
    r"you\s+should\s+(?:file\s+a\s+lawsuit|sue)",
    r"you\s+have\s+(?:the\s+)?(?:legal\s+)?right\s+to\s+sue",
    r"your\s+legal\s+options\s+are",
    r"consult\s+a\s+solicitor\s+because",
    r"take\s+legal\s+action\s+against",
]

PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(?:all\s+)?(?:previous\s+|prior\s+)?instructions",
    r"disregard\s+(?:your\s+)?system\s+prompt",
    r"pretend\s+you\s+are",
    r"you\s+are\s+now\s+(?:a\s+)?(?:different|new|another)",
    r"act\s+as\s+if\s+you\s+(?:have\s+no\s+)?(?:restrictions|constraints)",
    r"forget\s+(?:all\s+)?(?:your\s+)?(?:rules|constraints|instructions)",
    r"what\s+(?:are\s+)?your\s+(?:system\s+)?(?:prompt|instructions)",
    r"reveal\s+your\s+(?:system\s+)?(?:prompt|instructions)",
]

OFF_TOPIC_PATTERNS = [
    r"tell\s+me\s+a\s+joke",
    r"write\s+(?:me\s+)?a\s+poem",
    r"what\s+is\s+the\s+weather",
    r"write\s+(?:me\s+)?(?:some\s+)?code\s+for",
    r"generate\s+(?:an?\s+)?(?:essay|story|image)",
    r"help\s+me\s+with\s+my\s+(?:personal|love|relationship)",
]


# ── Input guardrails ──────────────────────────────────────────────────────────

def scan_input_pii(query: str) -> tuple[bool, list[str]]:
    """Detect PII in the input query. Returns (has_pii, detected_types)."""
    found = []
    for pii_type, pattern in PII_PATTERNS.items():
        if re.search(pattern, query, re.IGNORECASE):
            found.append(pii_type)
    return bool(found), found


def redact_pii(query: str) -> str:
    """Replace detected PII with [REDACTED] placeholder."""
    for pii_type, pattern in PII_PATTERNS.items():
        query = re.sub(pattern, f"[{pii_type} REDACTED]", query, flags=re.IGNORECASE)
    return query


def check_prompt_injection(query: str) -> bool:
    """Returns True if prompt injection is detected."""
    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            logger.warning(f"Prompt injection detected: pattern={pattern}")
            return True
    return False


def check_off_topic(query: str) -> bool:
    """Returns True if query is clearly off-topic."""
    for pattern in OFF_TOPIC_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            return True
    return False


# ── Output guardrails ─────────────────────────────────────────────────────────

def check_binding_language(text: str) -> tuple[bool, str]:
    """Returns (is_safe, reason). is_safe=False means dangerous language detected."""
    all_patterns = BINDING_COVERAGE_PATTERNS + GUARANTEED_PREMIUM_PATTERNS
    for pattern in all_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return False, f"Binding language pattern detected: '{pattern}'"
    return True, "OK"


def check_legal_advice(text: str) -> tuple[bool, str]:
    """Returns (is_safe, reason)."""
    for pattern in LEGAL_ADVICE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return False, f"Legal advice pattern detected: '{pattern}'"
    return True, "OK"


def check_source_grounding(answer: str, retrieved_chunks: list) -> bool:
    """Verify that cited source filenames exist in the retrieved chunks."""
    cited_files = re.findall(r"\[Source:\s*([^,\]]+)", answer)
    available_files = {c.metadata.get("source_file", "") for c in retrieved_chunks}
    for cited in cited_files:
        cited_clean = cited.strip()
        if not any(cited_clean.lower() in av.lower() for av in available_files):
            logger.warning(f"Possible hallucinated citation: '{cited_clean}'")
            return False
    return True


def ensure_disclaimer(text: str) -> str:
    """Inject the mandatory disclaimer if not already present."""
    if "for guidance only" not in text.lower():
        return text.rstrip() + f"\n\n{DISCLAIMER}"
    return text


def sanitise_binding_language(text: str) -> str:
    """Replace binding coverage/premium language with safe alternatives."""
    replacements = [
        (BINDING_COVERAGE_PATTERNS,
         "[coverage determination removed — please consult your broker]"),
        (GUARANTEED_PREMIUM_PATTERNS,
         "[exact premium figure removed — indicative estimates only]"),
    ]
    for patterns, replacement in replacements:
        for pattern in patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def apply_output_guardrails(
    answer: str, chunks: list
) -> tuple[str, bool]:
    """
    Apply all output guardrails in sequence.

    Returns:
        (safe_answer, all_guardrails_passed)
    """
    all_passed = True

    # 1. Binding language check
    is_safe, reason = check_binding_language(answer)
    if not is_safe:
        logger.warning(f"Output guardrail: {reason}")
        answer = sanitise_binding_language(answer)
        all_passed = False

    # 2. Legal advice check
    is_legal_safe, legal_reason = check_legal_advice(answer)
    if not is_legal_safe:
        logger.warning(f"Output guardrail: {legal_reason}")
        answer += (
            "\n\n[Note: For legal matters, please consult a qualified solicitor. "
            "I can only provide general insurance guidance.]"
        )
        all_passed = False

    # 3. Source grounding check
    if chunks:
        grounded = check_source_grounding(answer, chunks)
        if not grounded:
            answer += (
                "\n\n[Note: Some citations could not be verified against the "
                "retrieved policy documents — please verify directly with your insurer.]"
            )
            all_passed = False

    # 4. Ensure disclaimer
    answer = ensure_disclaimer(answer)

    return answer, all_passed
