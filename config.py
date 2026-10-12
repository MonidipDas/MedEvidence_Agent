"""
Configuration for the Self-Correcting Medical Evidence Agent.

Key resolution order for SARVAM_API_KEY:
  1. Streamlit secrets  (st.secrets["SARVAM_API_KEY"])  – used when deployed on Streamlit Cloud
  2. Environment variable (SARVAM_API_KEY)               – used locally or on other cloud hosts
  3. Hardcoded fallback                                   – development convenience only
"""
import os


def _resolve_api_key() -> str:
    """Resolve SarvamAI API key from st.secrets, env var, or fallback."""
    # 1. Try Streamlit secrets (available when deployed on Streamlit Cloud)
    try:
        import streamlit as st
        key = st.secrets.get("SARVAM_API_KEY", "")
        if key:
            return key
    except Exception:
        pass  # Not in a Streamlit context, or secrets not configured

    # 2. Try environment variable
    key = os.getenv("SARVAM_API_KEY", "")
    if key:
        return key

    # 3. Hardcoded fallback (development only)
    return "sk_7zzgz0xm_TqTNIr5cqMY0cSA1GupaHbk6"


# ─── SarvamAI API ───────────────────────────────────────────────────────────
SARVAM_API_KEY = _resolve_api_key()
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
