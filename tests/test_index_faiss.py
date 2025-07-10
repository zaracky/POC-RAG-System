import pytest
from index_faiss.build import build_vectorstore_from_docs
from langchain.schema import Document

def test_build_vectorstore_from_docs():
    docs = [
        Document(page_content="La nuit des musées à Paris", metadata={"source": "OpenAgenda"})
    ]
    vectordb = build_vectorstore_from_docs(docs)
    assert vectordb is not None

    results = vectordb.similarity_search("musées Paris")
    assert isinstance(results, list)
    assert len(results) > 0
    assert "musées" in results[0].page_content
