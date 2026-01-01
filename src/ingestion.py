from __future__ import annotations

import os
import logging
from pathlib import Path
import typer
from rich.logging import RichHandler
from rich.console import Console
from llama_index.core import SimpleDirectoryReader
from qdrant_client import QdrantClient, AsyncQdrantClient
from dotenv import load_dotenv

load_dotenv()

# Configure Rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("ingestion")
console = Console()

class Ingestion:
    
    def __init__(
        self,
        path: str = "data/"
    ) -> None:
        logger.info("[bold blue]Initializing Ingestion client...[/]", extra={"markup": True})
        
        qdrant_url = os.getenv("QDRANT_API")
        qdrant_key = os.getenv("QDRANT_API_KEY")
        
        if not qdrant_url or not qdrant_key:
            logger.error("[bold red]Missing QDRANT_API or QDRANT_API_KEY in environment variables.[/]", extra={"markup": True})
            raise ValueError("Qdrant credentials not found.")

        self._sync_client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
        self._async_client = AsyncQdrantClient(url=qdrant_url, api_key=qdrant_key)
        
        self.path = path
        self._validate_path()
        logger.info(f"Ingestion initialized with data path: [bold cyan]{self.path}[/]", extra={"markup": True})

    def _validate_path(self):
        path_obj = Path(self.path)
        if not path_obj.is_dir():
            error_msg = f"The path '{self.path}' is not a valid directory."
            logger.error(f"[bold red]{error_msg}[/]", extra={"markup": True})
            raise typer.BadParameter(error_msg)
        
       
        has_files = any(entry.is_file() for entry in path_obj.iterdir() if not entry.name.startswith('.'))
        if not has_files:
            error_msg = f"The directory '{self.path}' does not contain any data to ingest."
            logger.warning(f"[bold yellow]{error_msg}[/]", extra={"markup": True})
            raise typer.BadParameter(error_msg)
    
    def ingest(self):
        logger.info(f"Starting ingestion from [bold cyan]{self.path}[/]...", extra={"markup": True})
        try:
            reader = SimpleDirectoryReader(self.path)
            docs = reader.load_data()
            logger.info(f"Successfully loaded [bold green]{len(docs)}[/] documents.", extra={"markup": True})
            return docs
        except Exception as e:
            logger.error(f"[bold red]Error during ingestion: {e}[/]", extra={"markup": True})
            raise
