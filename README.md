# 🧬 Self-Correcting Medical Evidence Agent

A **self-correcting biomedical evidence agent** that retrieves PubMed evidence, generates answers using SarvamAI's SLM, verifies each claim via **DeBERTa NLI**, and automatically corrects unsupported statements. Built with **LangGraph** for orchestration and **Streamlit** for the UI.

> ⚠️ **Disclaimer**: This is a research prototype. Not intended for clinical decision-making.

---

## 🏗️ Architecture

```
                    ┌──────────────────────────────────────────────────┐
                    │              LangGraph State Machine              │
                    ├──────────────────────────────────────────────────┤
                    │                                                   │
    User Query ──▶  │  📚 Retrieve Evidence (PubMed E-utils)            │
                    │         │                                         │
                    │         ▼                                         │
                    │  🚫 Abstention Check (insufficient evidence?)     │
                    │      │ yes → ABSTAIN                              │
                    │      │ no ↓                                       │
                    │  🤖 Generate Answer (SarvamAI SLM)               │
                    │         │                                         │
                    │         ▼                                         │
                    │  🔍 Extract Atomic Claims (SarvamAI)             │
                    │         │                                         │
                    │         ▼                                         │
                    │  🧪 Verify Claims (DeBERTa NLI)                  │
                    │      │                                            │
                    │      ├─ faithful? → 📋 Final Output              │
                    │      │                                            │
                    │      └─ unfaithful? → 🔄 Self-Correct ──┐       │
                    │                              ↑           │       │
                    │                              └───────────┘       │
                    │                         (max 2 rounds)            │
                    └──────────────────────────────────────────────────┘
```

## 📦 Stack

| Component | Technology |
|---|---|
| **Evidence Retrieval** | PubMed NCBI E-utilities (BioASQ-style) |
| **Language Model** | SarvamAI SLM (`sarvam-m4`) |
| **Claim Verification** | DeBERTa v3 NLI (`cross-encoder/nli-deberta-v3-base`) |
| **Agent Orchestration** | LangGraph (StateGraph) |
| **Frontend** | Streamlit |
| **Visualization** | Plotly |

## 🔬 Research Metrics

The agent measures three key dimensions:

1. **Faithfulness** – Percentage of generated claims that are supported by retrieved evidence (NLI entailment score ≥ 0.65)
2. **Unsafe Claims** – Claims flagged as contradicted or unsupported by evidence
3. **Appropriate Abstention** – Agent refuses to answer when insufficient evidence is found

---

## 🚀 Setup Instructions

### Prerequisites

- **Python 3.10+** installed
- **pip** package manager
- Internet connection (for PubMed API & SarvamAI API & model download)

### Step 1: Create a virtual environment (recommended)

```bash
cd "c:\Users\MONIDIP DAS\OneDrive\Desktop\new project"

# Create virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Activate it (macOS/Linux)
# source venv/bin/activate
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

> **Note**: The first run will download the DeBERTa NLI model (~400MB). This is a one-time download.

### Step 3: Set your API key (optional – already configured)

The SarvamAI API key is pre-configured in `config.py`. To use a different key:

```bash
# Option A: Environment variable
set SARVAM_API_KEY=your_api_key_here       # Windows CMD
$env:SARVAM_API_KEY="your_api_key_here"    # Windows PowerShell
export SARVAM_API_KEY=your_api_key_here    # Linux/macOS

# Option B: Edit config.py directly
```

### Step 4: Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 📁 Project Structure

```
new project/
├── app.py              # Streamlit UI (frontend)
├── agent.py            # LangGraph agent orchestration
├── retriever.py        # PubMed biomedical RAG retriever
├── nli_checker.py      # DeBERTa NLI claim verification
├── llm_client.py       # SarvamAI LLM client
├── config.py           # Configuration & thresholds
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## 🧪 How to Use

1. **Enter a biomedical question** in the input field
   - e.g., "What is the efficacy of metformin for type 2 diabetes?"
2. Click **🔬 Analyze**
3. The agent will:
   - Retrieve relevant PubMed articles
   - Generate an evidence-based answer
   - Extract and verify each claim
   - Self-correct if needed
4. Review the results:
   - **Metrics dashboard** – Faithfulness, evidence count, claim verdicts
   - **Final answer** – Self-corrected, evidence-based response
   - **Claim verification** – Each claim with NLI scores
   - **Evidence passages** – Source articles from PubMed
   - **NLI charts** – Visual distribution of entailment/contradiction scores
   - **Execution trace** – Step-by-step agent log

## ⚙️ Configuration

Edit `config.py` to adjust:

| Parameter | Default | Description |
|---|---|---|
| `NLI_ENTAILMENT_THRESHOLD` | 0.65 | Min NLI score to consider a claim supported |
| `NLI_CONTRADICTION_THRESHOLD` | 0.55 | Min score to flag as contradicted |
| `ABSTENTION_MIN_EVIDENCE` | 1 | Min passages required before abstaining |
| `MAX_SELF_CORRECTION_ROUNDS` | 2 | Max self-correction iterations |
| `PUBMED_MAX_RESULTS` | 5 | Max articles to retrieve per query |

---

## 📊 Research Angles

### Measuring Faithfulness
The faithfulness score is computed as:
```
faithfulness = supported_claims / total_claims
```
Where a claim is "supported" if the DeBERTa NLI model assigns an entailment score ≥ 0.65 against any retrieved passage.

### Detecting Unsafe Claims
Claims are categorized as:
- **SUPPORTED** (✅) – Entailment ≥ 0.65
- **CONTRADICTED** (❌) – Contradiction ≥ 0.55
- **NEUTRAL** (🔶) – Weak entailment (0.3–0.65)
- **UNSUPPORTED** (⚠️) – No passage supports the claim

### Appropriate Abstention
The agent abstains when:
- Fewer than `ABSTENTION_MIN_EVIDENCE` passages are retrieved
- This prevents generating medical claims without evidence backing

---

## 🛡️ Safety Notice

This tool is designed for **biomedical research** purposes only. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare providers for medical decisions.
