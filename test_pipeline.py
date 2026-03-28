#!/usr/bin/env python3
"""
Manual integration test — exercises the full pipeline:
  1. Health check
  2. Document ingestion
  3. Due diligence analysis queries

Run after `docker compose up`:
  python test_pipeline.py
"""

import asyncio
import json
from pathlib import Path

import httpx

BASE_URL = "http://localhost:8000/api/v1"


async def check(label: str, response: httpx.Response):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Status: {response.status_code}")
    print(f"{'='*60}")
    try:
        data = response.json()
        print(json.dumps(data, indent=2))
    except Exception:
        print(response.text)
    return response


async def main():
    async with httpx.AsyncClient(timeout=120) as client:

        # ── 1. Health check ──────────────────────────────────────────────
        r = await client.get(f"{BASE_URL}/health")
        await check("HEALTH CHECK", r)

        # ── 2. Ingest document ───────────────────────────────────────────
        doc_content = Path("sample_docs/acme_cleantech_2024.txt").read_text()

        r = await client.post(f"{BASE_URL}/documents", json={
            "doc_id": "acme_2024_annual",
            "content": doc_content,
            "metadata": {
                "company": "Acme Cleantech Solutions",
                "year": 2024,
                "type": "annual_report"
            }
        })
        await check("INGEST DOCUMENT", r)

        # ── 3. Due diligence queries ─────────────────────────────────────

        queries = [
            "What is the customer concentration risk and how exposed is the company to losing its top customers?",
            "Assess the company's financial runway and burn rate. Is there a near-term liquidity risk?",
            "What are the main competitive threats and how defensible is this business?",
        ]

        for query in queries:
            r = await client.post(f"{BASE_URL}/analyze", json={
                "query": query,
                "doc_ids": ["acme_2024_annual"],
                "max_reasoning_steps": 4,
            })
            await check(f"ANALYSIS: {query[:50]}...", r)


if __name__ == "__main__":
    asyncio.run(main())
