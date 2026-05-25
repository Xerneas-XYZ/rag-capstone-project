import logging
from app.guardrails.rails import (
    check_binding_language,
    check_legal_advice,
    sanitise_binding_language,
    ensure_disclaimer,
)

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "This information is for guidance only. "
    "Please consult your broker or insurer for binding coverage confirmation."
)


def compliance_check(answer: str) -> tuple[str, bool]:
    """
    Run comprehensive compliance checks on the generated answer.

    Checks:
    1. Binding coverage language (e.g., "you are covered")
    2. Illegal legal advice (e.g., "you should sue")
    3. Ensures disclaimer is present

    Args:
        answer: The generated response text.

    Returns:
        Tuple of (sanitized_answer, is_compliant).
        - sanitized_answer: Answer with problematic language removed/replaced.
        - is_compliant: True if answer passed all checks; False if violations were sanitized.
    """
    is_compliant = True
    modified_answer = answer

    # ── Check 1: Binding coverage language ────────────────────────────
    safe_binding, reason_binding = check_binding_language(modified_answer)
    if not safe_binding:
        logger.warning(f"Binding language detected: {reason_binding}")
        is_compliant = False
        modified_answer = sanitise_binding_language(modified_answer)

    # ── Check 2: Legal advice ─────────────────────────────────────────
    safe_legal, reason_legal = check_legal_advice(modified_answer)
    if not safe_legal:
        logger.warning(f"Legal advice detected: {reason_legal}")
        is_compliant = False
        # For legal advice, we could sanitize or just log; for now, log and sanitize
        # You could replace these with warnings or placeholders
        modified_answer = modified_answer.replace(
            "you should", "[recommendation removed]"
        ).replace(
            "you have the right to sue", "[legal guidance removed]"
        )

    # ── Check 3: Ensure disclaimer ────────────────────────────────────
    if "for guidance only" not in modified_answer.lower():
        logger.info("Adding mandatory disclaimer")
        modified_answer = ensure_disclaimer(modified_answer)

    if is_compliant:
        logger.info("✓ Compliance check passed")
    else:
        logger.info("✓ Compliance check: violations found and sanitized")

    return modified_answer, is_compliant
