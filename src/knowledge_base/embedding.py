import os
import uuid
from dotenv import load_dotenv
import chromadb
from chromadb import Schema, SparseVectorIndexConfig, K
from chromadb.utils.embedding_functions import (
    ChromaCloudSpladeEmbeddingFunction,
)
from src.knowledge_base.ingestion_and_chunking import Ingestion

load_dotenv()

class Embedding:
    def __init__(self):
        self.ingestion = Ingestion()
        self.client = chromadb.CloudClient(
            api_key=os.getenv("CHROMA_API_KEY"),
            tenant=os.getenv("CHROMA_TENANT"),
            database=os.getenv("CHROMA_DATABASE"),
        )

        self._sparse_embedding_function = ChromaCloudSpladeEmbeddingFunction()

        schema = Schema()
        schema.create_index(
            config=SparseVectorIndexConfig(
                source_key=K.DOCUMENT,
                embedding_function=self._sparse_embedding_function,
            ),
            key="sparse_embedding",
        )

        self.collection = self.client.get_or_create_collection(
            name="rag_data",
            schema=schema,
        )

    def upload_files(self, files, admin_id):
        docs = self.ingestion.upload_files(files)

        file_ids = {}
        ids, texts, metadatas = [], [], []
        for doc in docs:
            source = doc.metadata["source"]
            file_id = file_ids.setdefault(source, str(uuid.uuid4()))
            chunk_id = str(uuid.uuid4())

            metadata = {
                "source": source,
                "pages": ",".join(str(p) for p in doc.metadata["pages"]),
                "admin_id": admin_id,
                "file_id": file_id,
                "chunk_id": chunk_id,
            }
            if doc.metadata.get("section"):
                metadata["section"] = doc.metadata["section"]

            ids.append(chunk_id)
            texts.append(doc.page_content)
            metadatas.append(metadata)

        for i in range(0, len(ids), 100):
            self.collection.add(
                ids=ids[i : i + 100],
                documents=texts[i : i + 100],
                metadatas=metadatas[i : i + 100],
            )

        return ids