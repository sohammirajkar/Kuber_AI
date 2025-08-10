from backend.main import TinyVectorStore

def test_vector_store_add_and_query():
    vs = TinyVectorStore(dim=32)
    id1 = vs.add("hello world", metadata={"text": "hello world"})
    # add another doc
    id2 = vs.add("another document", metadata={"text": "another document"})
    results = vs.query("hello", top_k=1)
    assert len(results) == 1
    assert "text" in results[0]
