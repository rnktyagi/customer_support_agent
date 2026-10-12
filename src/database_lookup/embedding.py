import os
from dotenv import load_dotenv
import chromadb
from chromadb import Schema, SparseVectorIndexConfig, K
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
    ChromaCloudSpladeEmbeddingFunction,
)
from src.database_lookup.scheme import Scheme

load_dotenv()


class SchemeEmbedding:
    def __init__(self):
        self.client = chromadb.CloudClient(
            api_key=os.getenv("CHROMA_API_KEY"),
            tenant=os.getenv("CHROMA_TENANT"),
            database=os.getenv("CHROMA_DATABASE"),
        )

        schema = Schema()
        schema.create_index(
            config=SparseVectorIndexConfig(
                source_key=K.DOCUMENT,
                embedding_function=ChromaCloudSpladeEmbeddingFunction(),
            ),
            key="sparse_embedding",
        )

        self.collection = self.client.get_or_create_collection(
            name="Text2SQL",
            schema=schema,
            embedding_function=SentenceTransformerEmbeddingFunction(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            ),
        )

    def create_embeddings(self, connection_url: str, admin_id: str):
        scheme = Scheme(connection_url, admin_id)
        docs = scheme.create_scheme()

        ids, texts, metadatas = [], [], []
        for doc in docs:
            ids.append(doc.metadata["table_id"])
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)

        for i in range(0, len(ids), 100):
            self.collection.add(
                ids=ids[i : i + 100],
                documents=texts[i : i + 100],
                metadatas=metadatas[i : i + 100],
            )

        return ids