
"""
One-time script to build the FAISS vector index from policy PDFs.
Run: python ingest/build_index.py
"""


import os
from pathlib import Path
from dotenv import load_dotenv
import glob
import json
import uuid
import sys

from langchain_core.documents import Document
# from docling.chunking import HybridChunker
# from transformers import AutoTokenizer, AutoModel
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType


from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import PdfFormatOption

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from app.config import settings



## To map the documents to their respective categories for better retrieval and organization in the vector store.
DOC_MAP = {
    "HomePolicy_COMPREHENSIVE_HomeShield": "comprehensive",
    "HomePolicy_LANDLORD_HomeShield": "landlord",
    "HomePolicy_STANDARD_HomeShield": "standard",
    "ClaimsProcedure_EvidenceGuide_HomeShield": "claims_procedure",
    "Contents_Valuables_CoverageGuide_HomeShield": "coverage_guide",
    "FullHomePolicy_Terms_HomeShield": "home_policy_terms",
    "HomeInsurance_Glossary_HomeShield": "home_insurance_glossary",
    "Perils_Exclusions_ReferenceGuide_HomeShield": "reference_guide"    
}

## To map the policy types to their respective policy categories for better retrieval and organization in the vector store.
POLICY_MAP = {
    "HomePolicy_COMPREHENSIVE_HomeShield": "comprehensive",
    "HomePolicy_LANDLORD_HomeShield": "landlord",
    "HomePolicy_STANDARD_HomeShield": "standard"
}


def infer_doc_type(filename: str) -> str:
    """Matches the raw file name identity to its designated document classification."""
    filename = str(filename)
    for key, dtype in DOC_MAP.items():
        if key in filename:
            return dtype
    return "general_home_doc"

def infer_tier(text: str) -> str:
    """Dynamically parses raw markdown strings to assign matching residential insurance tiers."""
    text = str(text)
    # text_lower = text.lower()
    for key, dtype in POLICY_MAP.items():
        if key in text:
            return dtype
    return "none"

def build_docling_doc_convertor():
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True

    converter = DocumentConverter()
    return converter


def build_index():
    pdf_dir = Path("data/pdfs")
    if not pdf_dir.exists():
        print(f"ERROR: PDF directory not found: {pdf_dir}")
        print("Ensure you have placed the dataset PDFs in data/pdfs/")
        sys.exit(1)
    
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"ERROR: No PDF files found in {pdf_dir}")
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDF files to process.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    if not settings.openai_api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Create a .env file in the project root or set the environment variable OPENAI_API_KEY."
        )

    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )

    all_chunks : list[Document] = []
    converter = build_docling_doc_convertor()
    for path in pdf_files:
        print(f"Processing: {path}")

        try:
            
            # Run the Docling processing pipeline
            result = converter.convert(path)
            
            # Export structural output to Markdown (highly accurate for tables/layouts)
            markdown_output = result.document.export_to_markdown()

            print(markdown_output)

            # Infer macro category metadata boundaries
            doc_type = infer_doc_type(path.name)
            
            # 2. Divide continuous layout structures into semantic chunks
            text_documents = splitter.create_documents(
                texts=[markdown_output],
                metadatas=[{"source_file": str(path)}]
            )
            
            # 3. Apply operational context schemas into chunk metadata layouts
            for doc in text_documents:
                doc.metadata.update({
                    "doc_type": doc_type,
                    "policy_tier": infer_tier(path.name),
                    "chunk_id": str(uuid.uuid4())
                })
                all_chunks.append(doc)
            
            print(f"    → {len(text_documents)} chunks (doc_type={doc_type})")
        except Exception as e:
            print(f"  ERROR processing {path}: {e}")


    print(f"\nTotal chunks to embed: {len(all_chunks)}")
    if not all_chunks:
        raise RuntimeError(
            "No valid document chunks were generated. Check that PDF conversion succeeded and that source documents produced text."
        )
    print("Building FAISS index (this may take a few minutes)...")

    # Path(settings.faiss_index_path).mkdir(parents=True, exist_ok=True)

    index = FAISS.from_documents(documents=all_chunks, embedding=embeddings)

    output_path = settings.faiss_index_path
    Path(settings.faiss_index_path).mkdir(parents=True, exist_ok=True)
    index.save_local(output_path)

    print(f"\nFAISS index saved to: {output_path}")
    print(f"Total vectors: {index.index.ntotal}")
    print("Ingestion complete.")



if __name__ == "__main__":
    build_index()


