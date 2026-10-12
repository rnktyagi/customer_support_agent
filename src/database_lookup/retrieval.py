import os
from dotenv import load_dotenv
import chromadb
from chromadb import Search, K, Knn, Rrf
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

load_dotenv()

class SchemeRetriever:
    def __init__(self):
        self.client = chromadb.CloudClient(
            api_key=os.getenv("CHROMA_API_KEY"),
            tenant=os.getenv("CHROMA_TENANT"),
            database=os.getenv("CHROMA_DATABASE"),
        )
        self.collection = self.client.get_collection(
            name="Text2SQL",
            embedding_function=SentenceTransformerEmbeddingFunction(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            ),
        )

    def retrieve(self, query, admin_id, top_k=5):
        hybrid_rank = Rrf(
            ranks=[
                Knn(query=query, return_rank=True, limit=50),
                Knn(query=query, key="sparse_embedding", return_rank=True, limit=50),
            ],
            weights=[0.7, 0.3],
            k=50,
        )

        search = (
            Search()
            .where(K("admin_id") == admin_id)
            .rank(hybrid_rank)
            .limit(top_k)
            .select_all()
        )

        rows = self.collection.search(search).rows()[0]
        if not rows:
            return ""

        dialect = rows[0]["metadata"]["dialect"]
        tables = "\n".join(row["metadata"]["table_info"] for row in rows)

        return f"Dialect : {dialect}\n{tables}\n"