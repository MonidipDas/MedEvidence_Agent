"""
Self-Correcting Medical Evidence Agent – Streamlit UI
=====================================================
A biomedical RAG agent that retrieves evidence from PubMed, generates
evidence-based answers, verifies claims via DeBERTa NLI, and self-corrects
unsupported claims. Abstains when evidence is insufficient.

Stack: BioASQ-style RAG · DeBERTa NLI · SarvamAI SLM · LangGraph
"""
import streamlit as st
import time
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from agent import run_agent

# ═══════════════════════════════════════════════════════════════════════════
# Page config & theme
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MedEvidence Agent · Self-Correcting Biomedical AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════
# Custom CSS
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global ── */
* { font-family: 'Inter', sans-serif; }
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #111827 40%, #0f172a 100%);
}

/* ── Header area ── */
.hero-container {
    background: linear-gradient(135deg, rgba(16,185,129,0.08) 0%, rgba(59,130,246,0.08) 50%, rgba(139,92,246,0.08) 100%);
    border: 1px solid rgba(16,185,129,0.15);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(20px);
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #10b981, #3b82f6, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    color: #94a3b8;
    font-size: 1rem;
    font-weight: 400;
    line-height: 1.6;
}
.hero-badge {
    display: inline-block;
    background: rgba(16,185,129,0.15);
    color: #10b981;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 8px;
    margin-top: 10px;
    border: 1px solid rgba(16,185,129,0.25);
}
.hero-badge-blue {
    background: rgba(59,130,246,0.15);
    color: #3b82f6;
    border-color: rgba(59,130,246,0.25);
}
.hero-badge-purple {
    background: rgba(139,92,246,0.15);
    color: #8b5cf6;
    border-color: rgba(139,92,246,0.25);
}

/* ── Cards ── */
.glass-card {
    background: rgba(30,41,59,0.6);
    border: 1px solid rgba(148,163,184,0.1);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(16px);
    transition: all 0.3s ease;
}
.glass-card:hover {
    border-color: rgba(16,185,129,0.3);
    box-shadow: 0 8px 32px rgba(16,185,129,0.08);
}
.card-title {
    color: #e2e8f0;
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Passage cards ── */
.passage-card {
    background: rgba(15,23,42,0.7);
    border: 1px solid rgba(59,130,246,0.15);
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 0.8rem;
    border-left: 3px solid #3b82f6;
    transition: all 0.2s ease;
}
.passage-card:hover {
    border-left-color: #10b981;
    background: rgba(15,23,42,0.9);
}
.passage-source {
    color: #64748b;
    font-size: 0.78rem;
    font-weight: 500;
    margin-top: 8px;
}
.passage-title {
    color: #93c5fd;
    font-weight: 600;
    font-size: 0.9rem;
    margin-bottom: 6px;
}
.passage-text {
    color: #cbd5e1;
    font-size: 0.83rem;
    line-height: 1.65;
}

/* ── Verdict badges ── */
.verdict-supported {
    background: rgba(16,185,129,0.12);
    border: 1px solid rgba(16,185,129,0.3);
    color: #34d399;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
}
.verdict-contradicted {
    background: rgba(239,68,68,0.12);
    border: 1px solid rgba(239,68,68,0.3);
    color: #fca5a5;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
}
.verdict-unsupported {
    background: rgba(245,158,11,0.12);
    border: 1px solid rgba(245,158,11,0.3);
    color: #fbbf24;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
}
.verdict-neutral {
    background: rgba(139,92,246,0.12);
    border: 1px solid rgba(139,92,246,0.3);
    color: #c4b5fd;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
}
.verdict-label {
    font-weight: 700;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}
.verdict-claim {
    font-size: 0.88rem;
    line-height: 1.5;
}
.verdict-meta {
    font-size: 0.75rem;
    opacity: 0.7;
    margin-top: 4px;
}

/* ── Metric cards ── */
.metric-card {
    background: rgba(30,41,59,0.5);
    border: 1px solid rgba(148,163,184,0.1);
    border-radius: 14px;
    padding: 1.2rem;
    text-align: center;
}
.metric-value {
    font-size: 2rem;
    font-weight: 800;
    margin-bottom: 4px;
}
.metric-label {
    color: #94a3b8;
    font-size: 0.8rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ── Trace log ── */
.trace-entry {
    background: rgba(15,23,42,0.5);
    border-left: 2px solid #3b82f6;
    padding: 0.5rem 1rem;
    margin-bottom: 4px;
    border-radius: 0 8px 8px 0;
    color: #cbd5e1;
    font-size: 0.82rem;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
}

/* ── Answer box ── */
.answer-box {
    background: linear-gradient(135deg, rgba(16,185,129,0.06) 0%, rgba(59,130,246,0.06) 100%);
    border: 1px solid rgba(16,185,129,0.2);
    border-radius: 16px;
    padding: 1.5rem;
    color: #e2e8f0;
    font-size: 0.95rem;
    line-height: 1.8;
}

/* ── Sidebar styling ── */
[data-testid="stSidebar"] {
    background: rgba(15,23,42,0.95) !important;
    border-right: 1px solid rgba(148,163,184,0.08);
}
[data-testid="stSidebar"] .stMarkdown p {
    color: #94a3b8;
}

/* ── Input field ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: rgba(30,41,59,0.8) !important;
    border: 1px solid rgba(148,163,184,0.15) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: rgba(16,185,129,0.5) !important;
    box-shadow: 0 0 0 2px rgba(16,185,129,0.15) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.6rem 2rem !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(16,185,129,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(16,185,129,0.4) !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: rgba(30,41,59,0.5) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}

/* ── Divider ── */
hr { border-color: rgba(148,163,184,0.1) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(148,163,184,0.2); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# Hero Header
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🧬 MedEvidence Agent</div>
    <div class="hero-subtitle">
        Self-correcting biomedical evidence synthesis with NLI-based claim verification.
        Retrieves PubMed evidence, generates answers, verifies each claim via DeBERTa NLI,
        and automatically corrects unsupported statements.
    </div>
    <div>
        <span class="hero-badge">PubMed RAG</span>
        <span class="hero-badge-blue hero-badge">DeBERTa NLI</span>
        <span class="hero-badge-purple hero-badge">SarvamAI SLM</span>
        <span class="hero-badge">LangGraph</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Agent Configuration")
    st.markdown("---")

    st.markdown("**Research Metrics**")
    st.markdown("""
    This agent measures three key dimensions:
    - 🎯 **Faithfulness** – % of claims supported by evidence
    - 🛡️ **Unsafe Claims** – contradicted or unverifiable assertions
    - 🚫 **Abstention** – refusal when evidence is insufficient
    """)

    st.markdown("---")
    st.markdown("**How It Works**")
    st.markdown("""
    ```
    1. 📚 Retrieve evidence (PubMed)
    2. 🤖 Generate answer (SarvamAI)
    3. 🔍 Extract atomic claims
    4. 🧪 Verify via NLI (DeBERTa)
    5. 🔄 Self-correct if needed
    6. 📋 Output with safety score
    ```
    """)

    st.markdown("---")
    st.markdown("**Graph Architecture**")
    st.markdown("""
    ```
    START → retrieve → abstain?
      ↓ no                ↓ yes
    generate → claims → NLI
      ↓                   ↓
    [faithful?]      → correct ↩
      ↓ yes
    output → END
    ```
    """)

    st.markdown("---")
    st.markdown("**Example Questions**")
    example_questions = [
        "What is the efficacy of metformin for type 2 diabetes?",
        "Does aspirin prevent cardiovascular events?",
        "What are the side effects of statins?",
        "Is CRISPR effective for sickle cell disease treatment?",
        "What is the role of gut microbiome in depression?",
    ]
    for q in example_questions:
        if st.button(q, key=f"ex_{hash(q)}", use_container_width=True):
            st.session_state["query_input"] = q

    st.markdown("---")
    st.markdown(
        "<div style='color:#64748b;font-size:0.75rem;text-align:center'>"
        "Built with LangGraph · DeBERTa · SarvamAI<br>"
        "⚠️ For research purposes only – not for clinical decisions"
        "</div>",
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════════════
# Main Input
# ═══════════════════════════════════════════════════════════════════════════
col_input, col_btn = st.columns([5, 1])
with col_input:
    query = st.text_input(
        "Ask a biomedical question",
        placeholder="e.g., What is the efficacy of metformin for type 2 diabetes?",
        key="query_input",
        label_visibility="collapsed",
    )
with col_btn:
    run_clicked = st.button("🔬 Analyze", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# Agent execution & results
# ═══════════════════════════════════════════════════════════════════════════
if run_clicked and query:
    # ── Progress tracking ──
    progress_container = st.empty()
    with progress_container.container():
        st.markdown("""
        <div class="glass-card" style="text-align:center;padding:2rem">
            <div style="font-size:2.5rem;margin-bottom:0.5rem">🧬</div>
            <div style="color:#e2e8f0;font-size:1.1rem;font-weight:600">Processing your query...</div>
            <div style="color:#94a3b8;font-size:0.85rem;margin-top:0.5rem">
                Retrieving evidence → Generating answer → Verifying claims → Self-correcting
            </div>
        </div>
        """, unsafe_allow_html=True)

    start_time = time.time()

    try:
        result = run_agent(query)
        elapsed = time.time() - start_time
        progress_container.empty()

        # ── Metrics Row ──
        st.markdown("### 📊 Analysis Results")
        m1, m2, m3, m4, m5 = st.columns(5)

        faithfulness = result.get("faithfulness_score", 0.0)
        num_passages = len(result.get("passages", []))
        num_claims = len(result.get("claims", []))
        num_supported = len(result.get("supported_claims", []))
        num_unsupported = len(result.get("unsupported_claims", []))
        correction_rounds = result.get("correction_round", 0)
        abstained = result.get("abstained", False)

        faith_color = "#10b981" if faithfulness >= 0.8 else "#f59e0b" if faithfulness >= 0.5 else "#ef4444"

        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{faith_color}">{faithfulness:.0%}</div>
                <div class="metric-label">Faithfulness</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:#3b82f6">{num_passages}</div>
                <div class="metric-label">Evidence Sources</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:#10b981">{num_supported}</div>
                <div class="metric-label">Supported Claims</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:#ef4444">{num_unsupported}</div>
                <div class="metric-label">Unsafe Claims</div>
            </div>
            """, unsafe_allow_html=True)
        with m5:
            abstain_text = "YES" if abstained else "NO"
            abstain_color = "#ef4444" if abstained else "#10b981"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{abstain_color}">{abstain_text}</div>
                <div class="metric-label">Abstained</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"<div style='text-align:center;color:#64748b;font-size:0.8rem;margin-top:0.5rem'>"
                     f"⏱️ Completed in {elapsed:.1f}s · 🔄 {correction_rounds} correction round(s)</div>",
                     unsafe_allow_html=True)

        st.markdown("---")

        # ── Final Answer ──
        col_answer, col_analysis = st.columns([3, 2])

        with col_answer:
            st.markdown('<div class="card-title">📋 Final Answer</div>', unsafe_allow_html=True)
            final_answer = result.get("final_answer", "No answer generated.")
            st.markdown(f'<div class="answer-box">{final_answer}</div>', unsafe_allow_html=True)

            # ── Evidence Passages ──
            st.markdown("")
            st.markdown('<div class="card-title">📚 Evidence Passages</div>', unsafe_allow_html=True)
            passages = result.get("passages", [])
            if passages:
                for i, p in enumerate(passages):
                    abstract_preview = p.get("abstract", "")[:300]
                    if len(p.get("abstract", "")) > 300:
                        abstract_preview += "..."
                    st.markdown(f"""
                    <div class="passage-card">
                        <div class="passage-title">{i+1}. {p.get('title', 'Untitled')}</div>
                        <div class="passage-text">{abstract_preview}</div>
                        <div class="passage-source">
                            {p.get('source_label', f"PMID: {p.get('pmid', 'Unknown')}")}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No evidence passages retrieved.")

        with col_analysis:
            # ── NLI Verdicts ──
            st.markdown('<div class="card-title">🧪 Claim Verification (NLI)</div>', unsafe_allow_html=True)

            verdicts = result.get("verdicts", [])
            if verdicts:
                for v in verdicts:
                    label = v.get("label", "UNKNOWN")
                    css_class = {
                        "SUPPORTED": "verdict-supported",
                        "CONTRADICTED": "verdict-contradicted",
                        "UNSUPPORTED": "verdict-unsupported",
                        "NEUTRAL": "verdict-neutral",
                    }.get(label, "verdict-unsupported")

                    st.markdown(f"""
                    <div class="{css_class}">
                        <div class="verdict-label">{v.get('icon','')} {label}</div>
                        <div class="verdict-claim">{v.get('claim','')}</div>
                        <div class="verdict-meta">
                            Entailment: {v.get('entailment_score',0):.3f} ·
                            Contradiction: {v.get('contradiction_score',0):.3f} ·
                            {v.get('explanation','')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # ── NLI Score Distribution Chart ──
                st.markdown("")
                st.markdown('<div class="card-title">📈 NLI Score Distribution</div>', unsafe_allow_html=True)

                if verdicts:
                    claims_short = [v["claim"][:40] + "..." if len(v["claim"]) > 40 else v["claim"] for v in verdicts]
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        name="Entailment", x=claims_short,
                        y=[v["entailment_score"] for v in verdicts],
                        marker_color="#10b981",
                    ))
                    fig.add_trace(go.Bar(
                        name="Contradiction", x=claims_short,
                        y=[v["contradiction_score"] for v in verdicts],
                        marker_color="#ef4444",
                    ))
                    fig.add_trace(go.Bar(
                        name="Neutral", x=claims_short,
                        y=[v["neutral_score"] for v in verdicts],
                        marker_color="#8b5cf6",
                    ))
                    fig.update_layout(
                        barmode="group",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#94a3b8", size=10),
                        legend=dict(orientation="h", y=-0.3),
                        margin=dict(l=20, r=20, t=20, b=80),
                        height=320,
                        xaxis=dict(tickangle=-30, showgrid=False),
                        yaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.1)"),
                    )
                    # Add threshold line
                    fig.add_hline(y=0.65, line_dash="dash", line_color="#10b981",
                                  annotation_text="Support threshold", annotation_font_color="#10b981")
                    st.plotly_chart(fig, use_container_width=True)

                # ── Verdict Summary Donut ──
                labels_count = {}
                for v in verdicts:
                    l = v["label"]
                    labels_count[l] = labels_count.get(l, 0) + 1

                if labels_count:
                    fig_donut = go.Figure(data=[go.Pie(
                        labels=list(labels_count.keys()),
                        values=list(labels_count.values()),
                        hole=0.55,
                        marker=dict(colors=[
                            {"SUPPORTED": "#10b981", "CONTRADICTED": "#ef4444",
                             "UNSUPPORTED": "#f59e0b", "NEUTRAL": "#8b5cf6"}.get(l, "#64748b")
                            for l in labels_count.keys()
                        ]),
                        textinfo="label+percent",
                        textfont=dict(size=11, color="#e2e8f0"),
                    )])
                    fig_donut.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#94a3b8"),
                        margin=dict(l=20, r=20, t=30, b=20),
                        height=280,
                        showlegend=False,
                        annotations=[dict(
                            text=f"{faithfulness:.0%}", x=0.5, y=0.5,
                            font_size=28, font_color=faith_color,
                            showarrow=False, font_family="Inter",
                        )],
                    )
                    st.plotly_chart(fig_donut, use_container_width=True)

            else:
                if abstained:
                    st.warning("Agent abstained – no claims to verify.")
                else:
                    st.info("No claim verdicts available.")

        # ── Execution Trace ──
        st.markdown("---")
        with st.expander("🔍 Execution Trace (Agent Log)", expanded=False):
            trace = result.get("trace_log", [])
            for entry in trace:
                st.markdown(f'<div class="trace-entry">{entry}</div>', unsafe_allow_html=True)

        # ── Raw State (Debug) ──
        with st.expander("🛠️ Raw Agent State (Debug)", expanded=False):
            # Make a copy that's JSON serializable
            debug_state = {}
            for k, v in result.items():
                try:
                    json.dumps(v)
                    debug_state[k] = v
                except (TypeError, ValueError):
                    debug_state[k] = str(v)
            st.json(debug_state)

    except Exception as e:
        progress_container.empty()
        st.error(f"❌ Agent Error: {str(e)}")
        with st.expander("Stack Trace"):
            import traceback
            st.code(traceback.format_exc())

elif run_clicked and not query:
    st.warning("Please enter a biomedical question to analyze.")
else:
    # ── Landing state ──
    st.markdown("")
    col_l1, col_l2, col_l3 = st.columns(3)
    with col_l1:
        st.markdown("""
        <div class="glass-card" style="text-align:center">
            <div style="font-size:2rem;margin-bottom:0.5rem">📚</div>
            <div class="card-title" style="justify-content:center">Evidence Retrieval</div>
            <p style="color:#94a3b8;font-size:0.85rem">
                Searches PubMed for relevant biomedical literature using
                BioASQ-style query optimization.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col_l2:
        st.markdown("""
        <div class="glass-card" style="text-align:center">
            <div style="font-size:2rem;margin-bottom:0.5rem">🧪</div>
            <div class="card-title" style="justify-content:center">NLI Verification</div>
            <p style="color:#94a3b8;font-size:0.85rem">
                Each claim is verified against source passages using
                DeBERTa-based Natural Language Inference.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col_l3:
        st.markdown("""
        <div class="glass-card" style="text-align:center">
            <div style="font-size:2rem;margin-bottom:0.5rem">🔄</div>
            <div class="card-title" style="justify-content:center">Self-Correction</div>
            <p style="color:#94a3b8;font-size:0.85rem">
                Unsupported claims are automatically removed and the answer
                is rewritten to maximize faithfulness.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;color:#64748b;margin-top:2rem;font-size:0.85rem">
        Enter a biomedical question above and click <b>🔬 Analyze</b> to begin.
    </div>
    """, unsafe_allow_html=True)
