import os, pathlib
from typing import List, Optional

import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document        # << current import path
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

def make_smart_retriever(base_retriever, *, llm_model="gpt-4o", n_alts=5):
    """
    `base_retriever` is typically  db.as_retriever(search_kwargs={"k": 3})
    Returns a MultiQueryRetriever that unions `n_alts` GPT-4o rewrites.
    """
    llm = ChatOpenAI(model=llm_model)

    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template=(
            """You are an AI language model assistant. Your task is to generate five
            different versions of the given question to retrieve relevant documents from
            a vector database. By generating multiple perspectives on the question, your
            goal is to help the user overcome some of the limitations of the distance-based
            similarity search. Provide these alternative questions separated by newlines.
            Original question:  {question}"""
        )
    ).partial(n=n_alts)

    return MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
        prompt=QUERY_PROMPT
    )

def _load_file(path: str, splitter) -> List[Document]:
    """Return a list[Document] for *one* file."""
    ext = pathlib.Path(path).suffix.lower()

    if ext == ".pdf":
        docs = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                if page_text := page.extract_text():
                    docs.extend(
                        Document(page_content=chunk,
                                 metadata={"type": "pdf", "file": path, "page": i})
                        for i, chunk in enumerate(splitter.split_text(page_text), 1)
                    )
        return docs

    # everything else → treat as plain text
    text = pathlib.Path(path).read_text(encoding="utf-8", errors="ignore")
    return [
        Document(page_content=chunk,
                 metadata={"type": "text", "file": path})
        for chunk in splitter.split_text(text)
    ]

def build_agent_vector_store(
    doc_paths: List[str],
    persist_dir: str,
    *,
    chunk_size: int = 200,
    chunk_overlap: int = 40,
    embedder: Optional[OpenAIEmbeddings] = None
) -> Chroma:
    """
    Create **or reuse** a Chroma store that contains *every* file in `doc_paths`.
    Any path ending in '.pdf' is split page-wise; everything else is treated as
    plain UTF-8 text.
    """

    embedder = OpenAIEmbeddings(
    model="text-embedding-3-large",     # or "text-embedding-3-large"
    dimensions=1536                    # omit to keep full 3072 for 3-large
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap)

    # ── 0   sanity-check source files ───────────────────────────────────
    for p in doc_paths:
        if not pathlib.Path(p).exists():
            raise FileNotFoundError(f"Vector-store source missing: {p}")

    # ── 1   open (or create) the collection ─────────────────────────────
    vectordb = Chroma(
        persist_directory=persist_dir,
        collection_name=pathlib.Path(persist_dir).name,
        embedding_function=embedder,
    )

    # ── 2   (re-)populate if the collection is empty ────────────────────
    if len(vectordb._collection.get()["ids"]) == 0:
        docs = []
        for p in doc_paths:
            docs.extend(_load_file(p, splitter))

        if not docs:
            raise ValueError("No text extracted from source files — nothing to embed.")

        vectordb.add_documents(docs)
        vectordb.persist()                       # <<< important
        print(f"Populated new vector store with {len(docs)} chunks")
    else:
        print(f"Re-using existing vector store at {persist_dir}")

    return vectordb
