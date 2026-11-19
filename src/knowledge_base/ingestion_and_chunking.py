from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from transformers import AutoTokenizer
from langchain_core.documents import Document
import uuid
import tempfile
import os

class Ingestion:
    def __init__(self):
        self.converter = DocumentConverter()
        self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
        self.chunker = HybridChunker(tokenizer=self.tokenizer, max_tokens=600, merge_peers=True)

    def ingestion_and_chunking(self, stream, filename: str):
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
            while True:
                chunk = stream.read(8192)
                if not chunk:
                    break
                tmp.write(chunk)
            tmp_path = tmp.name

        try:
            result = self.converter.convert(tmp_path)
        finally:
            os.unlink(tmp_path)

        chunks = list(self.chunker.chunk(result.document))

        for chunk in chunks:
            chunk.text = self.chunker.contextualize(chunk)

        docs = []

        for chunk in chunks:
            pages = sorted({
                prov.page_no
                for item in chunk.meta.doc_items
                for prov in item.prov
            })

            chunk_id = str(uuid.uuid4())

            docs.append(
                Document(
                    page_content=chunk.text,
                    metadata={
                        "source": filename,
                        "pages": pages,
                        "section": chunk.meta.headings[0] if chunk.meta.headings else None,
                    }
                )
            )

        return docs

    def upload_files(self, files: list[tuple]):
        docs = []

        for stream, filename in files:
            try:
                docs.extend(self.ingestion_and_chunking(stream, filename))
            except Exception as e:
                print(f"Failed to ingest {filename}: {e}")

        return docs
