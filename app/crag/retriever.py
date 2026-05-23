

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from app.config import get_settings
from functools import lru_cache

settings = get_settings()


@lru_cache(maxsize=1)
def get_faiss_index() -> FAISS:
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )
    return FAISS.load_local(
        settings.faiss_index_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def retrieve_chunks(
    query: str,
    doc_type: str = None,
    policy_tier: str = None,
    k: int = 6,
) -> list:
    """
    Retrieve top-k chunks from FAISS with optional metadata filtering.

    Args:
        query:       The search query string.
        doc_type:    Filter by coverage type (comprehensive | landlord |property | standard | glossary | and others ).
        policy_tier: Filter by tier (landlord | comprehensive| standard | none).
        k:           Number of chunks to return.

    Returns:
        List of LangChain Document objects with metadata.
    """
    index = get_faiss_index()
    filter_dict = {}
    if doc_type:
        filter_dict["doc_type"] = doc_type.lower()
    if policy_tier:
        filter_dict["policy_tier"] = policy_tier.lower()

    if filter_dict:
        fetch_k = max(k * 10, 50)
        docs = index.similarity_search(query, k=k, filter=filter_dict, fetch_k=fetch_k)
    else:
        docs = index.similarity_search(query, k=k)

    return docs


#To test the retriever module independently, you can run this script directly. Make sure to have the FAISS index built and the necessary environment variables set before running.
if __name__ == "__main__":
    print("Retriever module initialized .")
    print("FAISS index loaded and ready for retrieval.")

    index = get_faiss_index()
    filter_dict = {}
    
    filter_dict["doc_type"] = "general_home_doc"

    # filter_dict["policy_tier"] = "none"

    if filter_dict:
        docs = index.similarity_search("What does HomeShield cover?", k=5, filter=filter_dict)
    else:
        docs = index.similarity_search("What does HomeShield cover?", k=5)

    print(f"Retrieved {len(docs)} chunks:"  )
    print(docs)


    # filter_dict["policy_tier"] = "none"

    if filter_dict:
        docs = index.similarity_search("What does HomeShield cover?", k=5, filter=filter_dict)
    else:
        docs = index.similarity_search("What does HomeShield cover?", k=5)

    print(f"Retrieved {len(docs)} chunks with doc_type=coverage_guide and policy_tier=none:"  )
    print(docs)
    # return docs

    # print(retrieve_chunks("What does HomeShield cover?", doc_type="coverage_guide", policy_tier="comprehensive"))
    print("\n\n\n=================\n\n\n")
    print(retrieve_chunks("What is the claims procedure for HomeShield?"))