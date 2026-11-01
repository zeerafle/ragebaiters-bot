import os
import sqlite3
from datetime import date

import httpx
from pydantic import BaseModel
from pydantic_ai import Agent, Embedder, RunContext
from pydantic_ai.embeddings import EmbeddingSettings
from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

import store


class Posting(BaseModel):
    title: str
    organization: str
    url: str
    deadline: date | None
    location: str | None
    summary: str


provider = OpenAIProvider(
    base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"]
)
model = OpenAIChatModel(os.environ["CHAT_MODEL"], provider=provider)
embedder = Embedder(
    OpenAIEmbeddingModel(
        os.environ["EMBED_MODEL"],
        provider=provider,
        settings=EmbeddingSettings(dimensions=768),
    )
)


async def web_search(query: str, max_results: int = 8) -> list[dict]:
    """Search the web. Returns a list of {title, url, snippet}."""
    async with httpx.AsyncClient(timeout=20) as client:
        try:
            r = await client.post(
                f"{os.environ['LLM_BASE_URL']}/search",
                headers={"Authorization": f"Bearer {os.environ['LLM_API_KEY']}"},
                json={
                    "query": query,
                    "max_results": max_results,
                },
            )
            r.raise_for_status()  # 4xx/5xx don't raise on their own, so no fallback without this
        except httpx.HTTPError:
            r = await client.get(
                f"{os.environ['SEARXNG_URL']}/search",
                params={"q": query, "format": "json"},
            )
            r.raise_for_status()
    return [
        {
            "title": x["title"],
            "url": x["url"],
            "snippet": x.get("snippet") or x.get("content", ""),
        }
        for x in r.json()["results"][:max_results]
    ]


async def search_server(ctx: RunContext[sqlite3.Connection], query: str) -> str:
    """Search the Discord server's past messages. Returns the closest matches."""
    vec = (await embedder.embed_query(query)).embeddings[0]
    rows = store.search(ctx.deps, vec, 8)
    return (
        "\n".join(
            f"{r['author']} · {r['timestamp'][:10]} · {r['content']} ({r['url']})"
            for r in rows
        )
        or "No matching messages."
    )


chat_agent = Agent(
    model,
    deps_type=sqlite3.Connection,
    tools=[search_server, web_search],
    instructions=(
        "You are a casual Discord bot. Use search_server for anything about the "
        "server's past conversations and web_search for current facts. Cite message "
        "links. Say so when you don't know."
    ),
)
