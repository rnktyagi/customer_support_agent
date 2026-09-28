from fastembed.rerank.cross_encoder import TextCrossEncoder
from langchain_classic.retrievers import EnsembleRetriever  

def get_retrieve_client(vector_store, admin_id, top_k=5, bm25_retriever) :

    reranker=TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")

    vector_retriever=vector_store.as_retriever(search_kwargs={"k": top_k * 2, "filter": {"admin_id": admin_id}})

    bm25_retriever.k=top_k * 2

    ensemble_retriever=EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=[0.4, 0.6])

    return reranker, ensemble_retriever

def retrieve(query, admin_id, reranker, top_k=5, ensemble_retriever) :

    docs=ensemble_retriever.invoke(query)

    scores=list(reranker.rerank(query, [d.page_content for d in docs]))
    ranked=sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    return [doc for doc, _ in ranked[:top_k]]