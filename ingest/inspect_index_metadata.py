"""
Inspect FAISS index and list distinct `doc_type` and `policy_tier` values stored
in document metadata. Run: python ingest/inspect_index_metadata.py
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path so `app` package is importable when
# running this script directly.
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

def main():
    try:
        from app.crag.retriever import get_faiss_index
    except Exception as e:
        print("Failed to import retriever module:", repr(e))
        print("Ensure you're running from the project root and that dependencies and environment variables are set.")
        return

    idx = get_faiss_index()

    # Basic diagnostics
    try:
        ntotal = getattr(idx.index, "ntotal", None)
    except Exception:
        ntotal = None
    print("FAISS index ntotal:", ntotal)
    docstore = getattr(idx, "docstore", None)
    print("Docstore type:", type(docstore))

    doc_types = set()
    policy_tiers = set()

    # Prefer using index.index_to_docstore_id if available (preserves insertion order)
    ids = getattr(idx, "index_to_docstore_id", None)
    if ids:
        # show sample metadata for first few docs
        print('\nSample metadata for first 10 docs:')
        ids_list = list(ids)
        for i, _id in enumerate(ids_list[:10]):
            try:
                doc = idx.docstore.search(_id)
            except Exception:
                continue
            if not doc:
                continue
            md = getattr(doc, "metadata", {}) or {}
            print(i, _id, md)
            if "doc_type" in md:
                doc_types.add(md.get("doc_type"))
            if "policy_tier" in md:
                policy_tiers.add(md.get("policy_tier"))
    else:
        # Fallback: try to iterate over the docstore internal dict if present
        store_dict = getattr(idx.docstore, "_dict", None)
        if store_dict:
            for doc in store_dict.values():
                    md = getattr(doc, "metadata", {}) or {}
                    print('sample doc metadata:', md)
                    if "doc_type" in md:
                        doc_types.add(md.get("doc_type"))
                    if "policy_tier" in md:
                        policy_tiers.add(md.get("policy_tier"))

    print("Distinct doc_type values:")
    for v in sorted(x for x in doc_types if x is not None):
        print(" -", v)

    print("\nDistinct policy_tier values:")
    for v in sorted(x for x in policy_tiers if x is not None):
        print(" -", v)


if __name__ == "__main__":
    main()
