# rag_pipeline.py
from rag_utils import load_documents, split_documents, create_vector_store, create_rag_chain

if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)
    vectorstore = create_vector_store(chunks)
    rag_chain = create_rag_chain(vectorstore)

    print("RAG system is ready!")
    while True:
        q = input("Ask a question (or 'exit'): ")
        if q.lower() == "exit":
            break
        answer = rag_chain.run(q)
        print("Answer:", answer)
