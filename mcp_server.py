"""
MCP Server for the Self-Correcting Medical Evidence Agent.

Exposes three tools via the Model Context Protocol (MCP):
  • retrieve_evidence   – fetch PubMed passages for a query
  • verify_claim        – run DeBERTa NLI on a single claim vs passages
  • run_medical_agent   – execute the full self-correcting agent pipeline

Run this server standalone:
    python mcp_server.py

Then add it to your MCP client config (e.g. Claude Desktop):
    {
      "mcpServers": {
        "medevidence": {
          "command": "python",
          "args": ["path/to/mcp_server.py"]
        }
      }
    }
"""
import json
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp import types
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logger.warning("'mcp' package not installed. Run: pip install mcp")

from retriever import BiomedicalRetriever
from nli_checker import NLIClaimChecker
from agent import run_agent

_retriever = BiomedicalRetriever()
_nli = NLIClaimChecker()


# ══════════════════════════════════════════════════════════════════════════
# Tool definitions
# ══════════════════════════════════════════════════════════════════════════
TOOLS = [
    {
        "name": "retrieve_evidence",
        "description": (
            "Retrieve biomedical evidence passages from PubMed for a given question. "
            "Returns a list of relevant article abstracts with source metadata."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The biomedical question to retrieve evidence for.",
                }
            },
            "required": ["question"],
        },
    },
    {
        "name": "verify_claim",
        "description": (
            "Use DeBERTa NLI to verify whether a medical claim is supported, "
            "contradicted, or unsupported by a set of evidence passages."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "claim": {
                    "type": "string",
                    "description": "The atomic medical claim to verify.",
                },
                "passages": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of evidence passage texts to verify against.",
                },
            },
            "required": ["claim", "passages"],
        },
    },
    {
        "name": "run_medical_agent",
        "description": (
            "Run the full self-correcting medical evidence agent pipeline. "
            "Retrieves PubMed evidence, generates an evidence-based answer, "
            "extracts claims, verifies via NLI, self-corrects unsupported claims, "
            "and abstains if evidence is insufficient."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The biomedical question to answer.",
                }
            },
            "required": ["question"],
        },
    },
]


# ══════════════════════════════════════════════════════════════════════════
# Tool handlers
# ══════════════════════════════════════════════════════════════════════════
def handle_retrieve_evidence(args: dict) -> str:
    question = args["question"]
    passages = _retriever.retrieve(question)
    result = [
        {
            "pmid": p.pmid,
            "title": p.title,
            "abstract": p.abstract[:800],
            "source": p.source_label,
        }
        for p in passages
    ]
    return json.dumps({"passages": result, "count": len(result)}, indent=2)


def handle_verify_claim(args: dict) -> str:
    claim = args["claim"]
    passages = args["passages"]
    verdict = _nli.verify_claim(claim, passages)
    return json.dumps({
        "claim": verdict.claim,
        "label": verdict.label,
        "icon": verdict.icon,
        "confidence": round(verdict.confidence, 4),
        "entailment_score": round(verdict.entailment_score, 4),
        "contradiction_score": round(verdict.contradiction_score, 4),
        "neutral_score": round(verdict.neutral_score, 4),
        "explanation": verdict.explanation,
        "is_safe": verdict.is_safe,
    }, indent=2)


def handle_run_medical_agent(args: dict) -> str:
    question = args["question"]
    result = run_agent(question)
    return json.dumps({
        "final_answer": result.get("final_answer", ""),
        "faithfulness_score": round(result.get("faithfulness_score", 0.0), 4),
        "abstained": result.get("abstained", False),
        "correction_rounds": result.get("correction_round", 0),
        "num_passages": len(result.get("passages", [])),
        "num_claims": len(result.get("claims", [])),
        "supported_claims": result.get("supported_claims", []),
        "unsupported_claims": result.get("unsupported_claims", []),
        "verdicts": [
            {
                "claim": v["claim"],
                "label": v["label"],
                "icon": v["icon"],
                "entailment": round(v["entailment_score"], 4),
            }
            for v in result.get("verdicts", [])
        ],
        "trace_log": result.get("trace_log", []),
    }, indent=2)


HANDLERS = {
    "retrieve_evidence": handle_retrieve_evidence,
    "verify_claim": handle_verify_claim,
    "run_medical_agent": handle_run_medical_agent,
}


# ══════════════════════════════════════════════════════════════════════════
# MCP Server setup
# ══════════════════════════════════════════════════════════════════════════
async def main():
    if not MCP_AVAILABLE:
        print("ERROR: 'mcp' package not installed. Run: pip install mcp", file=sys.stderr)
        sys.exit(1)

    server = Server("medevidence-agent")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name=t["name"],
                description=t["description"],
                inputSchema=t["inputSchema"],
            )
            for t in TOOLS
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        handler = HANDLERS.get(name)
        if not handler:
            return [types.TextContent(type="text", text=f"Unknown tool: {name}")]
        try:
            result_text = handler(arguments)
            return [types.TextContent(type="text", text=result_text)]
        except Exception as e:
            logger.error("Tool '%s' error: %s", name, e, exc_info=True)
            return [types.TextContent(type="text", text=json.dumps({"error": str(e)}))]

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
