"""
ReAct Agent — Financial Due Diligence Analyst
----------------------------------------------
Implements the Reasoning + Acting (ReAct) pattern:

  Loop:
    1. THOUGHT  — the model reasons about what it knows and needs
    2. ACTION   — calls a tool (currently: retrieve_context)
    3. OBSERVATION — gets the tool result back
    4. Repeat until the model emits a FINAL_ANSWER

This keeps reasoning transparent and auditable — critical for financial
use cases where you need to show your work.

Design notes:
  - The agent is stateless per request; full history is passed each turn.
  - Tools are defined as plain async functions and described to the LLM
    in the system prompt. A production system would plug these into an
    MCP server so they're composable across agents.
  - We parse the LLM output with a lightweight text protocol instead of
    function-calling JSON, which works with any Ollama model.
"""

import json
import logging
import re
from typing import Any

import httpx

from app.config import settings
from app.rag.retriever import retrieve
from app.schemas import AnalysisResponse, ReasoningStep, RiskLevel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a senior financial due diligence analyst AI assistant.
Your job is to answer questions about financial documents using a structured
Reasoning + Acting loop.

AVAILABLE TOOLS:
  retrieve_context(query: str) → returns relevant passages from the document database

RESPONSE FORMAT — follow this EXACTLY for each reasoning step:

THOUGHT: <your reasoning about what you know and what you need>
ACTION: retrieve_context("<specific search query>")
OBSERVATION: <you will receive the tool result here>

... repeat THOUGHT/ACTION/OBSERVATION as needed (max {max_steps} steps) ...

When you have enough information, end with:

FINAL_ANSWER: <comprehensive answer>
RISK_LEVEL: <low|medium|high|critical>
KEY_FINDINGS:
- <finding 1>
- <finding 2>
- <finding 3>

Rules:
- Be specific in your retrieve_context queries — target exact financial metrics
- Always cite which document supports each claim
- If information is missing, state that explicitly in your risk assessment
- Never fabricate financial figures
"""

# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

async def _execute_tool(action_str: str, doc_ids: list[str] | None) -> str:
    """Parse an ACTION line and run the corresponding tool."""
    match = re.search(r'retrieve_context\(["\'](.+?)["\']\)', action_str)
    if not match:
        return "ERROR: Could not parse tool call. Use: retrieve_context(\"your query\")"

    query = match.group(1)
    chunks = await retrieve(query, top_k=4, doc_ids=doc_ids)

    if not chunks:
        return "No relevant passages found for this query."

    result_parts = []
    for chunk in chunks:
        source = f"[{chunk.doc_id} | chunk {chunk.metadata.get('chunk_index', '?')}]"
        result_parts.append(f"{source}\n{chunk.content}")

    return "\n\n---\n\n".join(result_parts)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

async def _call_llm(messages: list[dict]) -> str:
    """Call Ollama chat API and return the assistant message content."""
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{settings.ollama_base_url}/api/chat",
            json={
                "model": settings.chat_model,
                "messages": messages,
                "stream": False,
            },
        )
        response.raise_for_status()
        return response.json()["message"]["content"]


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def _parse_final_output(full_text: str) -> tuple[str, RiskLevel, list[str]]:
    """Extract FINAL_ANSWER, RISK_LEVEL, and KEY_FINDINGS from agent output."""

    answer_match = re.search(r"FINAL_ANSWER:\s*(.+?)(?=RISK_LEVEL:|$)", full_text, re.DOTALL)
    final_answer = answer_match.group(1).strip() if answer_match else "Analysis incomplete."

    risk_match = re.search(r"RISK_LEVEL:\s*(low|medium|high|critical)", full_text, re.IGNORECASE)
    risk_raw = risk_match.group(1).lower() if risk_match else "medium"
    risk_level = RiskLevel(risk_raw)

    findings_match = re.search(r"KEY_FINDINGS:\s*(.+?)$", full_text, re.DOTALL)
    key_findings: list[str] = []
    if findings_match:
        for line in findings_match.group(1).strip().splitlines():
            line = line.strip().lstrip("-•*").strip()
            if line:
                key_findings.append(line)

    return final_answer, risk_level, key_findings


def _parse_reasoning_steps(conversation: list[dict]) -> list[ReasoningStep]:
    """Reconstruct the reasoning steps from the agent's message history."""
    steps: list[ReasoningStep] = []
    step_num = 0

    for msg in conversation:
        if msg["role"] != "assistant":
            continue

        thought_matches = re.findall(r"THOUGHT:\s*(.+?)(?=ACTION:|FINAL_ANSWER:|$)", msg["content"], re.DOTALL)
        action_matches = re.findall(r"ACTION:\s*(.+?)(?=OBSERVATION:|THOUGHT:|FINAL_ANSWER:|$)", msg["content"], re.DOTALL)

        for i, thought in enumerate(thought_matches):
            step_num += 1
            action = action_matches[i].strip() if i < len(action_matches) else "none"

            # Find the corresponding observation from the next tool message
            observation = "(pending)"
            if i < len(conversation) - 1:
                next_msgs = [m for m in conversation if m["role"] == "tool"]
                if next_msgs:
                    observation = next_msgs[min(step_num - 1, len(next_msgs) - 1)]["content"][:300]

            steps.append(ReasoningStep(
                step=step_num,
                thought=thought.strip(),
                action=action,
                observation=observation,
            ))

    return steps


# ---------------------------------------------------------------------------
# Main agent entrypoint
# ---------------------------------------------------------------------------

async def run_analysis(
    query: str,
    doc_ids: list[str] | None = None,
    max_steps: int = 5,
) -> AnalysisResponse:
    """
    Run the ReAct due diligence agent and return a structured response.
    """
    system = SYSTEM_PROMPT.format(max_steps=max_steps)
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": query},
    ]

    tool_results: list[str] = []
    sources_used: list[str] = []
    full_agent_output = ""

    for step in range(max_steps):
        logger.info("[agent] step %d/%d", step + 1, max_steps)
        response_text = await _call_llm(messages)
        full_agent_output += response_text + "\n"

        messages.append({"role": "assistant", "content": response_text})

        # Check if the agent is done
        if "FINAL_ANSWER:" in response_text:
            logger.info("[agent] FINAL_ANSWER detected — stopping loop")
            break

        # Execute any ACTION in the response
        action_match = re.search(r"ACTION:\s*(.+?)(?=\n|$)", response_text)
        if action_match:
            action_str = action_match.group(1).strip()
            observation = await _execute_tool(action_str, doc_ids)
            tool_results.append(observation)

            # Extract source doc_ids from observation
            for doc_match in re.finditer(r"\[([^\]]+)\|", observation):
                sources_used.append(doc_match.group(1).strip())

            # Feed observation back as a user turn (simulating tool result)
            messages.append({
                "role": "user",
                "content": f"OBSERVATION:\n{observation}\n\nContinue your analysis.",
            })
        else:
            # No action found — ask the model to continue or conclude
            messages.append({
                "role": "user",
                "content": "Please continue. If you have enough information, provide your FINAL_ANSWER.",
            })

    # Parse the final structured output
    final_answer, risk_level, key_findings = _parse_final_output(full_agent_output)
    reasoning_steps = _parse_reasoning_steps(messages)

    # Deduplicate sources
    sources_used = list(dict.fromkeys(sources_used))

    return AnalysisResponse(
        query=query,
        reasoning_steps=reasoning_steps,
        final_answer=final_answer,
        risk_level=risk_level,
        key_findings=key_findings if key_findings else ["See final answer for details."],
        sources_used=sources_used,
    )
