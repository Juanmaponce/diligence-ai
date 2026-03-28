#!/usr/bin/env python3
"""
MCP Client Test — verifies the diligence-ai MCP server.

Connects via SSE to the running MCP server and exercises:
  · tools/list       — discover available tools
  · resources/list   — discover available resources
  · prompts/list     — discover available prompts
  · retrieve_context — RAG search tool
  · analyze_document — ReAct agent tool
  · documents://list — resource read

Run after `docker compose up`:
  pip install "mcp[cli]"
  python test_mcp_client.py
"""

import asyncio
import json

from mcp import ClientSession
from mcp.client.sse import sse_client

MCP_URL = "http://localhost:9000/sse"


async def section(title: str):
    print(f"\n{'━' * 60}")
    print(f"  {title}")
    print(f"{'━' * 60}")


async def main():
    print("\n🔌 Connecting to Diligence AI MCP server at", MCP_URL)

    async with sse_client(MCP_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("✓ Session initialized\n")

            # ── Discover tools ───────────────────────────────────────────
            await section("AVAILABLE TOOLS")
            tools = await session.list_tools()
            for tool in tools.tools:
                print(f"  🔧 {tool.name}")
                print(f"     {tool.description[:100]}...")
                print()

            # ── Discover resources ───────────────────────────────────────
            await section("AVAILABLE RESOURCES")
            resources = await session.list_resources()
            for resource in resources.resources:
                print(f"  📄 {resource.uri}  —  {resource.description[:80]}")

            # ── Discover prompts ─────────────────────────────────────────
            await section("AVAILABLE PROMPTS")
            prompts = await session.list_prompts()
            for prompt in prompts.prompts:
                print(f"  💬 {prompt.name}  —  {prompt.description[:80]}")

            # ── Read resource: documents://list ──────────────────────────
            await section("RESOURCE: documents://list")
            resource_result = await session.read_resource("documents://list")
            content = resource_result.contents[0].text
            data = json.loads(content)
            print(f"  Documents in knowledge base: {len(data.get('documents', []))}")
            for doc in data.get("documents", []):
                print(f"    · {doc['doc_id']}  ({doc['chunks']} chunks)")
            if not data.get("documents"):
                print("  (No documents yet — run test_pipeline.py first to ingest sample data)")

            # ── Tool: retrieve_context ───────────────────────────────────
            await section("TOOL: retrieve_context")
            print("  Query: 'customer concentration risk and top customers'\n")
            result = await session.call_tool(
                "retrieve_context",
                arguments={
                    "query": "customer concentration risk top customers ARR percentage",
                    "top_k": 3,
                    "doc_ids": ["acme_2024_annual"],
                },
            )
            print(result.content[0].text[:800])
            print("  ...")

            # ── Tool: analyze_document ───────────────────────────────────
            await section("TOOL: analyze_document (ReAct agent)")
            print("  Query: 'Is the financial runway sufficient and what are the burn risks?'\n")
            print("  (This will take ~30–90s depending on your hardware)\n")
            result = await session.call_tool(
                "analyze_document",
                arguments={
                    "query": "Is the financial runway sufficient? Assess burn rate and liquidity risk.",
                    "doc_ids": ["acme_2024_annual"],
                    "max_reasoning_steps": 3,
                },
            )
            print(result.content[0].text)

            # ── Prompt: due_diligence_brief ──────────────────────────────
            await section("PROMPT: due_diligence_brief")
            prompt_result = await session.get_prompt(
                "due_diligence_brief",
                arguments={"company_name": "Acme Cleantech Solutions", "focus_area": "risk"},
            )
            print(prompt_result.messages[0].content.text[:600])
            print("\n  ...")

            print(f"\n{'━' * 60}")
            print("  ✓ All MCP primitives verified successfully!")
            print(f"{'━' * 60}\n")


if __name__ == "__main__":
    asyncio.run(main())
