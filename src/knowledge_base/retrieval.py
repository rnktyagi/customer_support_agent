import os
from dotenv import load_dotenv
import chromadb
from chromadb import Search, K, Knn, Rrf
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from fastembed.rerank.cross_encoder import TextCrossEncoder
from langchain_core.documents import Document

load_dotenv()

class Retriever:
    def __init__(self):
        self.client = chromadb.CloudClient(
            api_key=os.getenv("CHROMA_API_KEY"),
            tenant=os.getenv("CHROMA_TENANT"),
            database=os.getenv("CHROMA_DATABASE"),
        )
        self.collection = self.client.get_collection(
            name="rag_data",
            embedding_function=SentenceTransformerEmbeddingFunction(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            ),
        )
        self.reranker = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")

    def retrieve(self, query, admin_id, top_k=5):
        hybrid_rank = Rrf(
            ranks=[
                Knn(query=query, return_rank=True, limit=100),
                Knn(query=query, key="sparse_embedding", return_rank=True, limit=100),
            ],
            weights=[0.7, 0.3],
            k=60,
        )

        search = (
            Search()
            .where(K("admin_id") == admin_id)
            .rank(hybrid_rank)
            .limit(top_k * 4)
            .select_all()
        )

        rows = self.collection.search(search).rows()[0]
        if not rows:
            return []

        docs = [
            Document(page_content=row["document"], metadata=row["metadata"])
            for row in rows
        ]

        scores = list(self.reranker.rerank(query, [d.page_content for d in docs]))
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in ranked[:top_k]]