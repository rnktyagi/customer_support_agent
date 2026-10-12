from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from transformers import AutoTokenizer
from langchain_core.documents import Document
from io import BytesIO
import uuid
from docling.datamodel.base_models import DocumentStream

class Ingestion:
    def __init__(self):
        self.converter = DocumentConverter()
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.chunker = HybridChunker(tokenizer=self.tokenizer, max_tokens=600, merge_peers=True)

    def ingestion_and_chunking(self, file_bytes: bytes, filename: str):
        source = DocumentStream(name=filename, stream=BytesIO(file_bytes))
        result = self.converter.convert(source)
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
                        "source": chunk.meta.origin.filename,
                        "pages": pages,
                        "section": chunk.meta.headings[0] if chunk.meta.headings else None,
                    }
                )
            )

        return docs

    def upload_files(self, files: list[tuple[bytes, str]]):
        docs = []

        for file_bytes, filename in files:
            try:
                docs.extend(self.ingestion_and_chunking(file_bytes, filename))
            except Exception as e:
                print(f"Failed to ingest {filename}: {e}")

        return docs
