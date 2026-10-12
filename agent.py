"""
LangGraph Agent – orchestrates the self-correcting medical evidence pipeline.

Graph flow:
  START → retrieve_evidence → generate_answer → extract_claims
        → verify_claims → [should_correct?]
            → yes → self_correct → extract_claims (loop)
            → no  → produce_output → END
"""
import operator
import logging
from typing import TypedDict, Annotated, Literal
from dataclasses import asdict

from langgraph.graph import StateGraph, START, END

from config import (
    ABSTENTION_MIN_EVIDENCE,
    MAX_SELF_CORRECTION_ROUNDS,
    CLAIM_EXTRACTION_TEMPERATURE,
    ANSWER_GENERATION_TEMPERATURE,
)
from llm_client import SarvamLLMClient
from retriever import BiomedicalRetriever, EvidencePassage
from nli_checker import NLIClaimChecker, ClaimVerdict

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Agent State
# ═══════════════════════════════════════════════════════════════════════════
class AgentState(TypedDict):
    """Shared state flowing through the LangGraph."""
    question: str
    passages: list[dict]            # serialised EvidencePassage dicts
    passage_texts: list[str]        # raw abstract texts for NLI
    answer: str                     # current generated answer
    claims: list[str]               # extracted atomic claims
    verdicts: list[dict]            # serialised ClaimVerdict dicts
    supported_claims: list[str]
    unsupported_claims: list[str]
    correction_round: int
    abstained: bool
    faithfulness_score: float       # % of supported claims
    final_answer: str
    trace_log: Annotated[list[str], operator.add]  # append-only event log


# ═══════════════════════════════════════════════════════════════════════════
# Node functions
# ═══════════════════════════════════════════════════════════════════════════
retriever = BiomedicalRetriever()
llm = SarvamLLMClient()
nli = NLIClaimChecker()


def retrieve_evidence(state: AgentState) -> dict:
    """Retrieve biomedical evidence passages from PubMed."""
    question = state["question"]
    passages: list[EvidencePassage] = retriever.retrieve(question)

    passage_dicts = [
        {
            "pmid": p.pmid, "title": p.title, "abstract": p.abstract,
            "journal": p.journal, "year": p.year, "authors": p.authors,
            "source_label": p.source_label,
        }
        for p in passages
    ]
    passage_texts = [p.full_text for p in passages]

    should_abstain = len(passages) < ABSTENTION_MIN_EVIDENCE

    log_entry = f"📚 Retrieved {len(passages)} evidence passages from PubMed"
    if should_abstain:
        log_entry += " → INSUFFICIENT EVIDENCE, will abstain"

    return {
        "passages": passage_dicts,
        "passage_texts": passage_texts,
        "abstained": should_abstain,
        "correction_round": 0,
        "trace_log": [log_entry],
    }


def check_abstention(state: AgentState) -> dict:
    """Check if we should abstain due to insufficient evidence."""
    if state["abstained"]:
        return {
            "answer": "",
            "claims": [],
            "verdicts": [],
            "supported_claims": [],
            "unsupported_claims": [],
            "faithfulness_score": 0.0,
            "final_answer": (
                "⚠️ **Abstention**: Insufficient biomedical evidence found to "
                "answer this question reliably. The agent chose to abstain rather "
                "than risk generating unsupported medical claims.\n\n"
                f"*Query*: {state['question']}\n"
                f"*Passages retrieved*: {len(state['passages'])}\n"
                f"*Minimum required*: {ABSTENTION_MIN_EVIDENCE}"
            ),
            "trace_log": ["🛑 ABSTAINED – insufficient evidence to generate a safe answer"],
        }
    return {"trace_log": ["✅ Sufficient evidence found, proceeding to answer generation"]}


def generate_answer(state: AgentState) -> dict:
    """Generate an evidence-based answer using the SLM."""
    if state["abstained"]:
        return {}

    passage_texts = state["passage_texts"]
    answer = llm.generate_answer(
        state["question"], passage_texts,
        temperature=ANSWER_GENERATION_TEMPERATURE,
    )
    round_num = state.get("correction_round", 0)
    label = "initial" if round_num == 0 else f"corrected (round {round_num})"
    return {
        "answer": answer,
        "trace_log": [f"🤖 Generated {label} answer ({len(answer)} chars)"],
    }


def extract_claims(state: AgentState) -> dict:
    """Extract atomic claims from the generated answer."""
    if state["abstained"]:
        return {}

    answer = state.get("answer", "").strip()
    if not answer:
        return {
            "claims": [],
            "trace_log": ["⚠️ Answer was empty – no claims to extract"],
        }

    claims = llm.extract_claims(answer)
    # Drop any blank claims that could crash the NLI model
    claims = [c for c in claims if c and c.strip()]
    return {
        "claims": claims,
        "trace_log": [f"🔍 Extracted {len(claims)} atomic claims from the answer"],
    }


def verify_claims(state: AgentState) -> dict:
    """Run NLI verification on each claim against evidence passages."""
    if state["abstained"]:
        return {}

    claims = state["claims"]
    passage_texts = state["passage_texts"]
    verdicts: list[ClaimVerdict] = nli.verify_all_claims(claims, passage_texts)

    supported = [v.claim for v in verdicts if v.label == "SUPPORTED"]
    unsupported = [v.claim for v in verdicts if v.label != "SUPPORTED"]

    total = len(verdicts) if verdicts else 1
    faithfulness = len(supported) / total

    verdict_dicts = [
        {
            "claim": v.claim, "label": v.label, "confidence": v.confidence,
            "best_passage_idx": v.best_passage_idx,
            "entailment_score": v.entailment_score,
            "contradiction_score": v.contradiction_score,
            "neutral_score": v.neutral_score,
            "explanation": v.explanation, "icon": v.icon,
        }
        for v in verdicts
    ]

    log_lines = [
        f"🧪 NLI Verification: {len(supported)}/{total} claims supported "
        f"(faithfulness: {faithfulness:.0%})",
    ]
    for v in verdicts:
        log_lines.append(f"   {v.icon} [{v.label}] {v.claim[:80]}...")

    return {
        "verdicts": verdict_dicts,
        "supported_claims": supported,
        "unsupported_claims": unsupported,
        "faithfulness_score": faithfulness,
        "trace_log": log_lines,
    }


def self_correct(state: AgentState) -> dict:
    """Rewrite the answer, removing unsupported claims."""
    corrected_answer = llm.rewrite_answer(
        question=state["question"],
        original_answer=state["answer"],
        supported_claims=state["supported_claims"],
        unsupported_claims=state["unsupported_claims"],
        context_passages=state["passage_texts"],
        temperature=ANSWER_GENERATION_TEMPERATURE,
    )
    new_round = state["correction_round"] + 1
    return {
        "answer": corrected_answer,
        "correction_round": new_round,
        "trace_log": [
            f"🔄 Self-correction round {new_round}: rewrote answer removing "
            f"{len(state['unsupported_claims'])} unsupported claims"
        ],
    }


def produce_output(state: AgentState) -> dict:
    """Produce the final output with metadata."""
    if state["abstained"]:
        return {}

    faithfulness = state["faithfulness_score"]
    answer = state["answer"]

    # Build safety notice
    if faithfulness >= 0.8:
        safety = "🟢 **High faithfulness** – most claims are evidence-supported."
    elif faithfulness >= 0.5:
        safety = "🟡 **Moderate faithfulness** – some claims may lack full support."
    else:
        safety = "🔴 **Low faithfulness** – several claims could not be verified. Interpret with caution."

    final = (
        f"{answer}\n\n---\n\n"
        f"**Faithfulness Score**: {faithfulness:.0%}\n"
        f"**Correction Rounds**: {state['correction_round']}\n"
        f"**Safety Assessment**: {safety}"
    )

    return {
        "final_answer": final,
        "trace_log": [f"📋 Final answer produced (faithfulness: {faithfulness:.0%})"],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Routing / conditional edges
# ═══════════════════════════════════════════════════════════════════════════
def should_continue_after_abstention(state: AgentState) -> Literal["generate", "output"]:
    """Route after abstention check."""
    if state["abstained"]:
        return "output"
    return "generate"


def should_correct(state: AgentState) -> Literal["correct", "output"]:
    """Decide whether to self-correct or produce final output."""
    if state["abstained"]:
        return "output"
    if (
        state["unsupported_claims"]
        and state["correction_round"] < MAX_SELF_CORRECTION_ROUNDS
        and state["faithfulness_score"] < 0.85
    ):
        return "correct"
    return "output"


# ═══════════════════════════════════════════════════════════════════════════
# Build the graph
# ═══════════════════════════════════════════════════════════════════════════
def build_agent_graph() -> StateGraph:
    """Construct and compile the LangGraph agent."""
    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("retrieve_evidence", retrieve_evidence)
    builder.add_node("check_abstention", check_abstention)
    builder.add_node("generate_answer", generate_answer)
    builder.add_node("extract_claims", extract_claims)
    builder.add_node("verify_claims", verify_claims)
    builder.add_node("self_correct", self_correct)
    builder.add_node("produce_output", produce_output)

    # Edges
    builder.add_edge(START, "retrieve_evidence")
    builder.add_edge("retrieve_evidence", "check_abstention")
    builder.add_conditional_edges(
        "check_abstention",
        should_continue_after_abstention,
        {"generate": "generate_answer", "output": "produce_output"},
    )
    builder.add_edge("generate_answer", "extract_claims")
    builder.add_edge("extract_claims", "verify_claims")
    builder.add_conditional_edges(
        "verify_claims",
        should_correct,
        {"correct": "self_correct", "output": "produce_output"},
    )
    builder.add_edge("self_correct", "extract_claims")   # Loop back
    builder.add_edge("produce_output", END)

    return builder.compile()


# Singleton compiled graph
agent_graph = build_agent_graph()


def run_agent(question: str) -> AgentState:
    """Execute the full agent pipeline for a medical question."""
    initial_state: AgentState = {
        "question": question,
        "passages": [],
        "passage_texts": [],
        "answer": "",
        "claims": [],
        "verdicts": [],
        "supported_claims": [],
        "unsupported_claims": [],
        "correction_round": 0,
        "abstained": False,
        "faithfulness_score": 0.0,
        "final_answer": "",
        "trace_log": [f"🚀 Agent started for question: {question}"],
    }
    result = agent_graph.invoke(initial_state)
    return result
