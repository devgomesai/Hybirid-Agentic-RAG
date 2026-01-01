from __future__ import annotations

import os
import logging
from rich.logging import RichHandler
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ConfigDict
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore

from src.ingestion import Ingestion
from src.prompt import SYSTEM_PROMPT
from src.vector_store import VectorDB

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("agent")


class RAGDependencies(BaseModel):
    """Dependencies injected into the RAG agent."""

    index: VectorStoreIndex = Field(
        ...,
        description="Vector index used to retrieve semantically similar documents",
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of top documents to retrieve per query",
    )
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )


model = OpenAIChatModel(
    model_name=os.getenv("LLM_MODEL", "gpt-4o-mini"),
    provider=OpenAIProvider(
        api_key=os.getenv("OPENAI_API_KEY")
    ),
)

agent = Agent[RAGDependencies, str](
    model=model,
    deps_type=RAGDependencies,
    system_prompt = SYSTEM_PROMPT,
)

# ------------------------------------------------------------------
# Tool: Hybrid Search
# ------------------------------------------------------------------
@agent.tool
async def search_documents(
    ctx: RunContext[RAGDependencies],
    query: str,
) -> str:
    """
    Search the vector database using hybrid (dense + BM25) retrieval.
    """
    logger.info(f"🔍 Searching documents for query: {query}")

    try:
        query_engine = ctx.deps.index.as_query_engine(
            vector_store_query_mode="hybrid",
            similarity_top_k=ctx.deps.top_k,
            sparse_top_k=ctx.deps.top_k,
        )

        response = query_engine.query(query)

        if not response.source_nodes:
            return "NO_RELEVANT_CONTEXT_FOUND"

        context_parts = []
        sources = []

        for i, node in enumerate(response.source_nodes, start=1):
            node: NodeWithScore

            text = node.node.get_content()
            metadata = node.node.metadata or {}

            source = metadata.get("file_name", f"Document {i}")
            sources.append(source)

            context_parts.append(
                f"--- SOURCE {i} ---\n"
                f"File: {source}\n"
                f"{text}\n"
            )

        # Deduplicate sources
        sources = list(dict.fromkeys(sources))

        logger.info(f"✅ Retrieved {len(sources)} sources")

        return (
            "RETRIEVED CONTEXT:\n\n"
            + "\n".join(context_parts)
            + "\n\nSOURCES:\n"
            + "\n".join(f"- {s}" for s in sources)
        )

    except Exception as e:
        logger.exception("❌ Error during document search")
        return f"ERROR_RETRIEVING_CONTEXT: {e}"


ingestor = Ingestion(path="data/")
documents = ingestor.ingest()

vector_store = VectorDB()

# Avoid re-indexing on every run
if not vector_store.client.collection_exists(vector_store.collection_name):
    logger.info("📦 Creating new vector index...")
    vector_store.create_index(documents)
else:
    logger.info("📂 Using existing vector index")

index = vector_store.get_index()

app = agent.to_web(
    deps=RAGDependencies(index=index, top_k=8),
    instructions="Think carefully about the user's intent, then retrieve context before answering.",
)

