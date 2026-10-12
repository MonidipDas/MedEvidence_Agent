"""
NLI Claim Checker – uses DeBERTa cross-encoder for Natural Language Inference.
Verifies each generated claim against source evidence passages.
"""
import logging
from dataclasses import dataclass
from config import NLI_MODEL_NAME, NLI_ENTAILMENT_THRESHOLD, NLI_CONTRADICTION_THRESHOLD

logger = logging.getLogger(__name__)

# Labels returned by the cross-encoder NLI model
NLI_LABELS = ["contradiction", "entailment", "neutral"]


@dataclass
class ClaimVerdict:
    """Verification result for a single claim."""
    claim: str
    label: str                  # "SUPPORTED", "CONTRADICTED", "UNSUPPORTED", "NEUTRAL"
    confidence: float           # Max NLI score across passages
    best_passage_idx: int       # Index of the passage with highest entailment
    entailment_score: float
    contradiction_score: float
    neutral_score: float
    explanation: str = ""

    @property
    def is_safe(self) -> bool:
        return self.label == "SUPPORTED"

    @property
    def icon(self) -> str:
        icons = {
            "SUPPORTED": "✅",
            "CONTRADICTED": "❌",
            "UNSUPPORTED": "⚠️",
            "NEUTRAL": "🔶",
        }
        return icons.get(self.label, "❓")


class NLIClaimChecker:
    """
    Uses DeBERTa-based cross-encoder to perform NLI between
    each claim and the source evidence passages.
    """

    def __init__(self, model_name: str = NLI_MODEL_NAME):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        """Lazy-load the cross-encoder model."""
        if self._model is None:
            logger.info("Loading NLI model: %s", self.model_name)
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            logger.info("NLI model loaded successfully")
        return self._model

    def verify_claim(self, claim: str, passages: list[str]) -> ClaimVerdict:
        """
        Verify a single claim against multiple evidence passages.
        Returns the aggregate verdict.
        """
        if not passages:
            return ClaimVerdict(
                claim=claim, label="UNSUPPORTED", confidence=0.0,
                best_passage_idx=-1, entailment_score=0.0,
                contradiction_score=0.0, neutral_score=0.0,
                explanation="No evidence passages available.",
            )

        model = self._load_model()

        # Create pairs: (passage, claim) for each passage
        pairs = [(passage, claim) for passage in passages]
        scores = model.predict(pairs)

        # scores shape: (num_passages, 3) for [contradiction, entailment, neutral]
        best_entailment = -1.0
        best_contradiction = -1.0
        best_passage_idx = 0
        best_scores = None

        for idx, score_row in enumerate(scores):
            # score_row = [contradiction, entailment, neutral]
            ent_score = float(score_row[1])  # entailment
            con_score = float(score_row[0])  # contradiction

            if ent_score > best_entailment:
                best_entailment = ent_score
                best_passage_idx = idx
                best_scores = score_row

            if con_score > best_contradiction:
                best_contradiction = con_score

        cont_score = float(best_scores[0])
        ent_score = float(best_scores[1])
        neut_score = float(best_scores[2])

        # Determine verdict
        if best_entailment >= NLI_ENTAILMENT_THRESHOLD:
            label = "SUPPORTED"
            confidence = best_entailment
            explanation = f"Claim is supported by passage {best_passage_idx + 1} (entailment: {best_entailment:.3f})"
        elif best_contradiction >= NLI_CONTRADICTION_THRESHOLD:
            label = "CONTRADICTED"
            confidence = best_contradiction
            explanation = f"Claim is contradicted by evidence (contradiction: {best_contradiction:.3f})"
        elif best_entailment > 0.3:
            label = "NEUTRAL"
            confidence = best_entailment
            explanation = f"Weak evidence – claim is partially related but not fully supported (entailment: {best_entailment:.3f})"
        else:
            label = "UNSUPPORTED"
            confidence = 1.0 - best_entailment
            explanation = f"No passage supports this claim (max entailment: {best_entailment:.3f})"

        return ClaimVerdict(
            claim=claim, label=label, confidence=confidence,
            best_passage_idx=best_passage_idx,
            entailment_score=ent_score,
            contradiction_score=cont_score,
            neutral_score=neut_score,
            explanation=explanation,
        )

    def verify_all_claims(self, claims: list[str], passages: list[str]) -> list[ClaimVerdict]:
        """Verify a list of claims against the evidence passages."""
        verdicts = []
        for claim in claims:
            verdict = self.verify_claim(claim, passages)
            verdicts.append(verdict)
            logger.info(
                "Claim: '%s...' → %s (%.3f)",
                claim[:60], verdict.label, verdict.confidence,
            )
        return verdicts
