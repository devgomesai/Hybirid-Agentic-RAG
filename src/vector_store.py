from __future__ import annotations

import os
import logging
from typing import List
from rich.logging import RichHandler
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, StorageContext, Settings, Document
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, AsyncQdrantClient

load_dotenv()

# Configure Rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("vector_store")

class VectorDB:
    def __init__(
        self,
        collection_name: str = "Hybrid_RAG",
        chunk_size: int = 512,
    ) -> None:
        logger.info(f"[bold blue]Initializing VectorDB with collection: {collection_name}[/]", extra={"markup": True})
        
        qdrant_url = os.getenv("QDRANT_API")
        qdrant_key = os.getenv("QDRANT_API_KEY")
        
        if not qdrant_url or not qdrant_key:
            logger.error("[bold red]Missing QDRANT_API or QDRANT_API_KEY environment variables.[/]", extra={"markup": True})
            raise ValueError("Qdrant credentials not found.")

        self.client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
        self.aclient = AsyncQdrantClient(url=qdrant_url, api_key=qdrant_key)
        self.collection_name = collection_name
        
        # Set global settings as per notebook
        Settings.chunk_size = chunk_size
        logger.info(f"Settings.chunk_size set to [bold cyan]{chunk_size}[/]", extra={"markup": True})

    def create_index(self, documents: List[Document]) -> None:
        """
        Creates a VectorStoreIndex from a list of documents using Qdrant.
        """
        logger.info(f"Setting up QdrantVectorStore for collection [bold green]{self.collection_name}[/]...", extra={"markup": True})
        
        try:
            vector_store = QdrantVectorStore(
                collection_name=self.collection_name,
                client=self.client,
                aclient=self.aclient,
                enable_hybrid=True,
                fastembed_sparse_model="Qdrant/bm25",
                batch_size=20,
            )

            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            logger.info(f"Indexing [bold cyan]{len(documents)}[/] documents... (This might take a while)", extra={"markup": True})
            
            VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                use_async=True,
                show_progress=True
            )
            
            logger.info("[bold green]Index successfully created![/]", extra={"markup": True})

        except Exception as e:
            logger.error(f"[bold red]Failed to create index: {e}[/]", extra={"markup": True})
            raise

    def get_index(self) -> VectorStoreIndex:
        """
        Loads an existing index from the vector store.
        """
        logger.info(f"Loading existing index from collection [bold green]{self.collection_name}[/]...", extra={"markup": True})
        try:
            vector_store = QdrantVectorStore(
                collection_name=self.collection_name,
                client=self.client,
                aclient=self.aclient,
                enable_hybrid=True,
                fastembed_sparse_model="Qdrant/bm25",
                batch_size=20,
            )
            
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
            )
            logger.info("[bold green]Index successfully loaded![/]", extra={"markup": True})
            return index
        except Exception as e:
            logger.error(f"[bold red]Failed to load index: {e}[/]", extra={"markup": True})
            raise
