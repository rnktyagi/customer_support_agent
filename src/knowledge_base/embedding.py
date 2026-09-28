import os
import uuid
from dotenv import load_dotenv
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.knowledge_base.ingestion_and_chunking import upload_files
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever  

load_dotenv()

BATCH_SIZE=100

def get_embedding_clients() :

    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    chroma_client=chromadb.CloudClient(
        api_key=os.getenv("CHROMA_API_KEY"),
        tenant=os.getenv("CHROMA_TENANT"),
        database=os.getenv("CHROMA_DATABASE", "default_database"),
        cloud_host=os.getenv("CHROMA_CLOUD_HOST"),
        )
    
    vector_store=Chroma.Chroma(embedding_function=embedding_model, client=chroma_client, collection_name="rag_data")

    return vector_store

def embedding_files(files: list[tuple[bytes, str]], converter, chunker, vector_store, admin_id) :
    docs = upload_files(files, converter, chunker)

    bm25_retriever=BM25Retriever.from_documents(docs)
    
    file_ids = {}
    ids = []
    for doc in docs:
        source = doc.metadata["source"]
        if source not in file_ids:
            file_ids[source] = str(uuid.uuid4())
 
        chunk_id = str(uuid.uuid4())
        doc.metadata["admin_id"] = admin_id
        doc.metadata["file_id"] = file_ids[source]
        doc.metadata["chunk_id"] = chunk_id
        ids.append(chunk_id)
 
    for i in range(0, len(docs), BATCH_SIZE):
        vector_store.add_documents(
            documents=docs[i:i + BATCH_SIZE],
            ids=ids[i:i + BATCH_SIZE],
        )
 
    return {
        "files_ingested": len(files),
        "chunks_added": len(docs)
    }
    
    