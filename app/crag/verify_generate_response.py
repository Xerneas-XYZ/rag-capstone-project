"""Simple verifier for `generate_response` in `app.crag.generator`.

This script monkeypatches the generation chain to avoid external LLM calls
and validates that `generate_response` formats context and forwards the
query. Run with:

    python app/crag/verify_generate_response.py

It will raise AssertionError on failure and print "OK" on success.
"""
import sys
from pathlib import Path
from types import SimpleNamespace


def main():
    # Ensure project root is importable when running this script directly
    root_dir = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(root_dir))

    import app.crag.generator as generator

    # Dummy result and chain to avoid external LLM calls
    class DummyResult:
        def __init__(self, content: str):
            self.content = content

    class DummyChain:
        def invoke(self, payload: dict):
            # basic payload validation
            if not isinstance(payload, dict):
                raise AssertionError("invoke payload must be a dict")
            if "query" not in payload:
                raise AssertionError("payload missing 'query' key")
            # return deterministic response including the query for verification
            return DummyResult(f"DUMMY RESPONSE for: {payload.get('query')}")

    # Monkeypatch the generator's chain with our dummy
    generator.generate_chain = DummyChain()

    query = "What does the HomeShield policy cover for valuables?"

    # Create fake chunks matching the minimal Document interface used by the generator
    chunks = [
        SimpleNamespace(
            metadata={
                "source_file": "HomePolicy_COMPREHENSIVE_HomeShield.pdf",
                "section_title": "Coverage",
                "page_num": 1,
            },
            page_content="Coverage: This policy covers X, Y, Z.",
        ),
    ]

    print(f"Using {len(chunks)} synthetic chunks for query: '{query}'")

    result = generator.generate_response(query, chunks, confidence="HIGH")

    print("generate_response returned:\n", result)

    # Simple assertions to verify behavior
    assert "DUMMY RESPONSE for:" in result, "generate_response did not return expected dummy response"
    assert query in result, "Returned response does not contain the query (expected in dummy response)"

    print("OK")


if __name__ == "__main__":
    main()
