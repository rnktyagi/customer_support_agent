from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from transformers import AutoTokenizer
from langchain_core.documents import Document
from io import BytesIO

def create_ingestion_clients() :
    converter = DocumentConverter()
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    chunker = HybridChunker(tokenizer=tokenizer, max_tokens=600 , merge_peers=True)

    return converter, chunker

def ingestion_and_chunking(file_bytes : bytes , filename : str, converter, chunker) :
    
    source=DocumentStream(name=filename, stream=BytesIO(file_bytes))
    result = converter.convert(source)
    chunks=list(chunker.chunk(result.document))

    for chunk in chunks :
        chunk.text=chunker.contextualize(chunk)
    
    docs=[]

    for chunk in chunks:
        pages = sorted({
            prov.page_no
            for item in chunk.meta.doc_items
            for prov in item.prov})
        
        chunk_id=str(uuid.uuid4())
 
        docs.append(
            Document(
                page_content=chunk.text,
                metadata={
                    "source": chunk.meta.origin.filename,
                    "pages": pages,
                    "section": chunk.meta.headings[0] if chunk.meta.headings else None,
                }
            )
        )
 
    return docs

def upload_files(files: list[tuple[bytes, str]], converter, chunker) :
    docs = []
 
    for file_bytes, filename in files :
        try :
            docs.extend(ingestion_and_chunking(file_bytes, filename, converter, chunker))
        except Exception as e :
            print(f"Failed to ingest {filename}: {e}")
 
    return docs