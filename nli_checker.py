"""
NLI Claim Checker – uses DeBERTa cross-encoder for Natural Language Inference.
Verifies each generated claim against source evidence passages.

Bug-fixes vs previous version:
  - Handle 1-D scores array (single passage → CrossEncoder returns 1-D, not 2-D)
  - Guard against empty / whitespace-only claims
  - Guard against best_scores staying None if scores iteration is empty
"""
import logging
import numpy as np
from dataclasses import dataclass
from config import NLI_MODEL_NAME, NLI_ENTAILMENT_THRESHOLD, NLI_CONTRADICTION_THRESHOLD

logger = logging.getLogger(__name__)


@dataclass
class ClaimVerdict:
    """Verification result for a single claim."""
    claim: str
    label: str          # "SUPPORTED" | "CONTRADICTED" | "UNSUPPORTED" | "NEUTRAL"
    confidence: float
    best_passage_idx: int
    entailment_score: float
    contradiction_score: float
    neutral_score: float
    explanation: str = ""

    @property
    def is_safe(self) -> bool:
        return self.label == "SUPPORTED"

    @property
    def icon(self) -> str:
        return {"SUPPORTED": "✅", "CONTRADICTED": "❌",
                "UNSUPPORTED": "⚠️", "NEUTRAL": "🔶"}.get(self.label, "❓")


class NLIClaimChecker:
    """DeBERTa-based cross-encoder NLI verification."""

    def __init__(self, model_name: str = NLI_MODEL_NAME):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            logger.info("Loading NLI model: %s", self.model_name)
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            logger.info("NLI model loaded")
        return self._model

    def verify_claim(self, claim: str, passages: list[str]) -> ClaimVerdict:
        """Verify a single claim against evidence passages."""
        # ── Guard: trivial / empty claim ──────────────────────────────────
        if not claim or not claim.strip():
            return ClaimVerdict(
                claim=claim, label="UNSUPPORTED", confidence=0.0,
                best_passage_idx=-1, entailment_score=0.0,
                contradiction_score=0.0, neutral_score=0.0,
                explanation="Empty or trivial claim – skipped.",
            )

        if not passages:
            return ClaimVerdict(
                claim=claim, label="UNSUPPORTED", confidence=0.0,
                best_passage_idx=-1, entailment_score=0.0,
                contradiction_score=0.0, neutral_score=0.0,
                explanation="No evidence passages available.",
            )

        model = self._load_model()

        # Build (premise, hypothesis) pairs
        pairs = [(p[:1024], claim[:512]) for p in passages]   # truncate to avoid OOM

        try:
            raw_scores = model.predict(pairs)
        except Exception as e:
            logger.error("CrossEncoder.predict failed: %s", e)
            return ClaimVerdict(
                claim=claim, label="UNSUPPORTED", confidence=0.0,
                best_passage_idx=-1, entailment_score=0.0,
                contradiction_score=0.0, neutral_score=0.0,
                explanation=f"NLI model error: {e}",
            )

        # ── Normalise to always 2-D: shape (n_passages, 3) ───────────────
        scores = np.array(raw_scores, dtype=float)
        if scores.ndim == 1:
            # Single passage → CrossEncoder may return shape (3,) not (1,3)
            scores = scores.reshape(1, -1)

        if scores.shape[0] == 0 or scores.shape[-1] < 3:
            logger.warning("Unexpected scores shape: %s", scores.shape)
            return ClaimVerdict(
                claim=claim, label="UNSUPPORTED", confidence=0.0,
                best_passage_idx=-1, entailment_score=0.0,
                contradiction_score=0.0, neutral_score=0.0,
                explanation="Unexpected NLI output shape.",
            )

        # scores columns: [contradiction=0, entailment=1, neutral=2]
        best_ent_idx = int(np.argmax(scores[:, 1]))
        best_scores_row = scores[best_ent_idx]

        cont_score = float(best_scores_row[0])
        ent_score  = float(best_scores_row[1])
        neut_score = float(best_scores_row[2])

        best_contradiction = float(np.max(scores[:, 0]))

        # ── Verdict ───────────────────────────────────────────────────────
        if ent_score >= NLI_ENTAILMENT_THRESHOLD:
            label = "SUPPORTED"
            confidence = ent_score
            explanation = (f"Supported by passage {best_ent_idx + 1} "
                           f"(entailment: {ent_score:.3f})")
        elif best_contradiction >= NLI_CONTRADICTION_THRESHOLD:
            label = "CONTRADICTED"
            confidence = best_contradiction
            explanation = f"Contradicted by evidence (contradiction: {best_contradiction:.3f})"
        elif ent_score > 0.3:
            label = "NEUTRAL"
            confidence = ent_score
            explanation = f"Weakly related but not fully supported (entailment: {ent_score:.3f})"
        else:
            label = "UNSUPPORTED"
            confidence = 1.0 - ent_score
            explanation = f"No passage supports this claim (max entailment: {ent_score:.3f})"

        return ClaimVerdict(
            claim=claim, label=label, confidence=confidence,
            best_passage_idx=best_ent_idx,
            entailment_score=ent_score,
            contradiction_score=cont_score,
            neutral_score=neut_score,
            explanation=explanation,
        )

    def verify_all_claims(self, claims: list[str], passages: list[str]) -> list[ClaimVerdict]:
        """Verify a list of claims; skips empty/whitespace-only claims."""
        verdicts = []
        for claim in claims:
            if not claim or not claim.strip():
                continue   # skip blank claims – the root cause of the NoneType crash
            verdict = self.verify_claim(claim, passages)
            verdicts.append(verdict)
            logger.info("Claim: '%.60s' → %s (%.3f)", claim, verdict.label, verdict.confidence)
        return verdicts
