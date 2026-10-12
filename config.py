"""
Configuration for the Self-Correcting Medical Evidence Agent.
"""
import os

# ─── SarvamAI API ───────────────────────────────────────────────────────────
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "sk_7zzgz0xm_TqTNIr5cqMY0cSA1GupaHbk6")
SARVAM_BASE_URL = "https://api.sarvam.ai"
SARVAM_MODEL = "sarvam-m4"  # SLM model

# ─── NLI Model (DeBERTa) ───────────────────────────────────────────────────
NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"

# ─── Retriever Settings ────────────────────────────────────────────────────
PUBMED_MAX_RESULTS = 5          # Max articles to retrieve per query
PUBMED_EMAIL = "agent@medicalevidence.ai"  # Required by NCBI E-utils

# ─── NLI Thresholds ────────────────────────────────────────────────────────
NLI_ENTAILMENT_THRESHOLD = 0.65   # Min score to consider a claim supported
NLI_CONTRADICTION_THRESHOLD = 0.55  # Min score to flag as contradicted
ABSTENTION_MIN_EVIDENCE = 1       # Min number of passages required

# ─── Agent Settings ────────────────────────────────────────────────────────
MAX_SELF_CORRECTION_ROUNDS = 2    # Max times the agent can self-correct
CLAIM_EXTRACTION_TEMPERATURE = 0.2
ANSWER_GENERATION_TEMPERATURE = 0.3
