import os
from typing import List

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# openai = Vercel-friendly (no torch). huggingface = local/Docker with sentence-transformers.
DEFAULT_EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "openai").strip().lower()
DEFAULT_OPENAI_EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
DEFAULT_HF_EMBEDDING_MODEL = os.environ.get("HF_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
# OpenAI text-embedding-3-* supports Matryoshka dims; 1536 is the full small model.
DEFAULT_EMBEDDING_DIMENSIONS = {
    "openai": int(os.environ.get("EMBEDDING_DIMENSIONS", "1536")),
    "huggingface": 384,
}


def load_pdf_file(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Keep only the metadata we need: the source file name and the 1-based page number
    (used to show citations like "Gale Encyclopedia, p. 1047").
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = os.path.basename(doc.metadata.get("source", "") or "")
        page = doc.metadata.get("page")
        meta = {"source": src}
        if isinstance(page, int):
            meta["page"] = page + 1
        minimal_docs.append(Document(page_content=doc.page_content, metadata=meta))
    return minimal_docs


def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(extracted_data)


def embedding_dimensions(provider: str | None = None) -> int:
    p = (provider or DEFAULT_EMBEDDING_PROVIDER).strip().lower()
    return DEFAULT_EMBEDDING_DIMENSIONS.get(p, 1536)


def get_embeddings(provider: str | None = None, model_name: str | None = None):
    """Return the LangChain embeddings client for the configured provider."""
    p = (provider or DEFAULT_EMBEDDING_PROVIDER).strip().lower()
    if p in {"openai", "openai_api"}:
        from langchain_openai import OpenAIEmbeddings

        kwargs = {
            "model": model_name or DEFAULT_OPENAI_EMBEDDING_MODEL,
            "api_key": os.environ.get("OPENAI_API_KEY"),
        }
        dims = embedding_dimensions("openai")
        if dims and dims != 1536:
            kwargs["dimensions"] = dims
        return OpenAIEmbeddings(**kwargs)

    if p in {"huggingface", "hf", "local"}:
        return download_hugging_face_embeddings(model_name)

    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER={p!r}. Use 'openai' (Vercel) or 'huggingface' (Docker/local ML)."
    )


def download_hugging_face_embeddings(model_name: str | None = None):
    """Local HuggingFace embeddings — requires torch + sentence-transformers (not for Vercel)."""
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model_name or DEFAULT_HF_EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
