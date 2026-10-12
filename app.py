"""
MedEvidence Agent – Streamlit UI (v2 · clean redesign)
=======================================================
Self-correcting biomedical evidence agent with NLI claim verification.
"""
import streamlit as st
import time
import plotly.graph_objects as go
from agent import run_agent

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedEvidence Agent",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal clean CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Layout ── */
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1200px; }

/* ── Header ── */
.header { margin-bottom: 1.5rem; }
.header h1 {
    font-size: 1.8rem; font-weight: 700; color: #f1f5f9;
    margin: 0 0 0.3rem 0; letter-spacing: -0.3px;
}
.header p { color: #64748b; font-size: 0.88rem; margin: 0; }
.badge {
    display: inline-block; font-size: 0.7rem; font-weight: 600;
    padding: 2px 10px; border-radius: 999px; margin-right: 6px; margin-top: 8px;
    letter-spacing: 0.3px;
}
.badge-green  { background: #052e16; color: #34d399; border: 1px solid #065f46; }
.badge-blue   { background: #0c1a3a; color: #60a5fa; border: 1px solid #1e3a8a; }
.badge-purple { background: #1e0a40; color: #c084fc; border: 1px solid #4c1d95; }
.badge-gray   { background: #1e293b; color: #94a3b8; border: 1px solid #334155; }

/* ── Metric strip ── */
.metric-strip {
    display: flex; gap: 12px; margin: 1rem 0;
}
.metric-box {
    flex: 1; background: #1e293b; border: 1px solid #334155;
    border-radius: 10px; padding: 14px 16px; text-align: center;
}
.metric-box .val { font-size: 1.7rem; font-weight: 700; margin-bottom: 2px; }
.metric-box .lbl { font-size: 0.72rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; }

/* ── Cards ── */
.card {
    background: #1e293b; border: 1px solid #334155;
    border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 0.75rem;
}
.card-header {
    font-size: 0.8rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.5px; color: #94a3b8; margin-bottom: 0.75rem;
}

/* ── Evidence passage ── */
.passage {
    border-left: 3px solid #3b82f6; padding: 0.7rem 1rem;
    background: #0f172a; border-radius: 0 8px 8px 0;
    margin-bottom: 0.6rem;
}
.passage-title { color: #93c5fd; font-size: 0.85rem; font-weight: 600; margin-bottom: 4px; }
.passage-body  { color: #cbd5e1; font-size: 0.8rem; line-height: 1.6; }
.passage-meta  { color: #475569; font-size: 0.72rem; margin-top: 6px; }

/* ── Verdict chips ── */
.verdict {
    border-radius: 8px; padding: 0.65rem 0.9rem;
    margin-bottom: 0.5rem; border-left: 3px solid;
}
.verdict .v-label { font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; }
.verdict .v-claim { font-size: 0.83rem; margin-top: 3px; line-height: 1.5; }
.verdict .v-meta  { font-size: 0.72rem; margin-top: 4px; opacity: 0.65; }

.v-supported    { background: #052e16; border-color: #16a34a; color: #86efac; }
.v-contradicted { background: #450a0a; border-color: #dc2626; color: #fca5a5; }
.v-unsupported  { background: #1c1003; border-color: #d97706; color: #fcd34d; }
.v-neutral      { background: #150b30; border-color: #7c3aed; color: #c4b5fd; }

/* ── Answer box ── */
.answer-box {
    background: #0f172a; border: 1px solid #1e3a5f;
    border-radius: 10px; padding: 1.2rem 1.4rem;
    color: #e2e8f0; font-size: 0.88rem; line-height: 1.8;
    white-space: pre-wrap;
}

/* ── Trace log ── */
.trace-line {
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.77rem; color: #94a3b8;
    border-left: 2px solid #334155; padding: 3px 10px;
    margin-bottom: 3px; background: #0f172a; border-radius: 0 4px 4px 0;
}

/* ── Safety pill ── */
.safety-pill {
    display: inline-block; font-size: 0.78rem; font-weight: 600;
    padding: 4px 14px; border-radius: 999px; margin-top: 10px;
}
.safety-high   { background: #052e16; color: #34d399; border: 1px solid #065f46; }
.safety-mid    { background: #1c1003; color: #fcd34d; border: 1px solid #92400e; }
.safety-low    { background: #450a0a; color: #fca5a5; border: 1px solid #7f1d1d; }
.safety-abs    { background: #1e0a40; color: #c084fc; border: 1px solid #4c1d95; }

/* ── Sidebar ── */
[data-testid="stSidebar"] { background: #0f172a !important; border-right: 1px solid #1e293b; }
[data-testid="stSidebar"] .stButton > button {
    background: #1e293b !important; color: #e2e8f0 !important;
    border: 1px solid #334155 !important; border-radius: 8px !important;
    font-size: 0.78rem !important; text-align: left !important;
    padding: 0.4rem 0.8rem !important; margin-bottom: 2px !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    border-color: #3b82f6 !important; color: #93c5fd !important;
    transform: none !important;
}

/* ── Main CTA button ── */
div[data-testid="stHorizontalBlock"] .stButton > button {
    background: linear-gradient(135deg, #1d4ed8 0%, #4338ca 100%) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
    font-size: 0.88rem !important; padding: 0.55rem 1.6rem !important;
    box-shadow: 0 2px 8px rgba(59,130,246,0.35) !important;
    transition: opacity 0.2s !important;
}
div[data-testid="stHorizontalBlock"] .stButton > button:hover {
    opacity: 0.88 !important; transform: none !important; box-shadow: none !important;
}

/* ── Input ── */
.stTextInput > div > div > input {
    background: #1e293b !important; border: 1px solid #334155 !important;
    border-radius: 8px !important; color: #e2e8f0 !important;
    font-size: 0.88rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.2) !important;
}
.stTextInput label { color: #64748b !important; font-size: 0.8rem !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0f172a !important; border-bottom: 1px solid #334155 !important; gap: 0;
}
.stTabs [data-baseweb="tab"] {
    color: #64748b !important; font-size: 0.82rem !important; font-weight: 500 !important;
    padding: 0.5rem 1.2rem !important; border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #60a5fa !important; border-bottom-color: #3b82f6 !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1rem; }

/* ── Expander ── */
.streamlit-expanderHeader { background: #1e293b !important; border-radius: 8px !important; }

/* ── Divider ── */
hr { border-color: #1e293b !important; margin: 1rem 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="header">
  <h1>🧬 MedEvidence Agent</h1>
  <p>Self-correcting biomedical evidence synthesis · NLI claim verification · Appropriate abstention</p>
  <div>
    <span class="badge badge-green">PubMed RAG</span>
    <span class="badge badge-blue">DeBERTa NLI</span>
    <span class="badge badge-purple">SarvamAI 105B</span>
    <span class="badge badge-gray">LangGraph</span>
    <span class="badge badge-gray">MCP</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("#### 🔬 Example Questions")

    examples = [
        "What is the efficacy of metformin for type 2 diabetes?",
        "Does aspirin prevent cardiovascular events?",
        "What are the side effects of statins?",
        "Is CRISPR effective for sickle cell disease?",
        "What is the role of gut microbiome in depression?",
        "How effective is immunotherapy for lung cancer?",
    ]
    for q in examples:
        if st.button(q, key=f"ex_{hash(q)}", use_container_width=True):
            st.session_state["query"] = q

    st.markdown("---")
    st.markdown("#### ⚙️ Pipeline")
    st.markdown("""
```
1  Retrieve (PubMed)
2  Generate answer (SarvamAI)
3  Extract claims
4  Verify via NLI (DeBERTa)
5  Self-correct if needed
6  Abstain if insufficient
```
""")

    st.markdown("---")
    st.markdown("#### 🔌 MCP Server")
    st.markdown("""
    Run `python mcp_server.py` to expose agent tools via **Model Context Protocol**:
    - `retrieve_evidence`
    - `verify_claim`
    - `run_medical_agent`
    """)

    st.markdown("---")
    st.caption("⚠️ Research prototype — not for clinical decisions.")


# ══════════════════════════════════════════════════════════════════════════
# Input row
# ══════════════════════════════════════════════════════════════════════════
col_q, col_btn = st.columns([5, 1])
with col_q:
    query = st.text_input(
        "Biomedical question",
        value=st.session_state.get("query", ""),
        placeholder="e.g. What is the efficacy of metformin for type 2 diabetes?",
        key="query",
    )
with col_btn:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    run = st.button("Analyze →", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# Run agent
# ══════════════════════════════════════════════════════════════════════════
if run and query:
    with st.spinner("Retrieving evidence, generating answer, verifying claims…"):
        t0 = time.time()
        try:
            result = run_agent(query)
            elapsed = time.time() - t0
        except Exception as e:
            st.error(f"**Agent Error**: {e}")
            import traceback
            with st.expander("Stack trace"):
                st.code(traceback.format_exc())
            st.stop()

    # ── pull fields ───────────────────────────────────────────────────────
    abstained       = result.get("abstained", False)
    faithfulness    = result.get("faithfulness_score", 0.0)
    passages        = result.get("passages", [])
    claims          = result.get("claims", [])
    verdicts        = result.get("verdicts", [])
    supported       = result.get("supported_claims", [])
    unsupported     = result.get("unsupported_claims", [])
    correction_rnd  = result.get("correction_round", 0)
    final_answer    = result.get("final_answer", "")
    trace           = result.get("trace_log", [])

    # ── metric strip ──────────────────────────────────────────────────────
    faith_color = "#34d399" if faithfulness >= 0.8 else "#fbbf24" if faithfulness >= 0.5 else "#f87171"
    abs_color   = "#f87171" if abstained else "#34d399"

    st.markdown(f"""
    <div class="metric-strip">
      <div class="metric-box">
        <div class="val" style="color:{faith_color}">{faithfulness:.0%}</div>
        <div class="lbl">Faithfulness</div>
      </div>
      <div class="metric-box">
        <div class="val" style="color:#60a5fa">{len(passages)}</div>
        <div class="lbl">Sources</div>
      </div>
      <div class="metric-box">
        <div class="val" style="color:#34d399">{len(supported)}</div>
        <div class="lbl">Supported</div>
      </div>
      <div class="metric-box">
        <div class="val" style="color:#f87171">{len(unsupported)}</div>
        <div class="lbl">Unsafe Claims</div>
      </div>
      <div class="metric-box">
        <div class="val" style="color:#c084fc">{correction_rnd}</div>
        <div class="lbl">Corrections</div>
      </div>
      <div class="metric-box">
        <div class="val" style="color:{abs_color}">{"YES" if abstained else "NO"}</div>
        <div class="lbl">Abstained</div>
      </div>
    </div>
    <div style="color:#475569;font-size:0.75rem;margin-bottom:1rem">
      ⏱ {elapsed:.1f}s
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── tabs ──────────────────────────────────────────────────────────────
    tab_ans, tab_nli, tab_src, tab_trace = st.tabs([
        "📋 Answer", "🧪 NLI Verification", "📚 Evidence Sources", "🔍 Trace"
    ])

    # ── Tab 1: Answer ─────────────────────────────────────────────────────
    with tab_ans:
        if abstained:
            st.warning(final_answer, icon="🚫")
        else:
            # Safety pill
            if faithfulness >= 0.8:
                pill_cls, pill_txt = "safety-high", "🟢 High Faithfulness"
            elif faithfulness >= 0.5:
                pill_cls, pill_txt = "safety-mid", "🟡 Moderate Faithfulness"
            else:
                pill_cls, pill_txt = "safety-low", "🔴 Low Faithfulness"

            st.markdown(
                f'<span class="safety-pill {pill_cls}">{pill_txt} · {faithfulness:.0%} of claims supported</span>',
                unsafe_allow_html=True,
            )
            st.markdown("")

            # Strip the metadata footer that's embedded in final_answer for display
            answer_body = result.get("answer", final_answer)
            st.markdown(f'<div class="answer-box">{answer_body}</div>', unsafe_allow_html=True)

    # ── Tab 2: NLI ───────────────────────────────────────────────────────
    with tab_nli:
        if not verdicts:
            st.info("No claim verdicts available.")
        else:
            col_v, col_chart = st.columns([3, 2])

            with col_v:
                for v in verdicts:
                    lbl = v.get("label", "UNSUPPORTED")
                    css = {
                        "SUPPORTED": "v-supported", "CONTRADICTED": "v-contradicted",
                        "UNSUPPORTED": "v-unsupported", "NEUTRAL": "v-neutral",
                    }.get(lbl, "v-unsupported")
                    icon = v.get("icon", "")
                    claim_txt = v.get("claim", "")
                    ent = v.get("entailment_score", 0)
                    con = v.get("contradiction_score", 0)
                    expl = v.get("explanation", "")
                    st.markdown(f"""
                    <div class="verdict {css}">
                      <div class="v-label">{icon} {lbl}</div>
                      <div class="v-claim">{claim_txt}</div>
                      <div class="v-meta">Entailment {ent:.3f} · Contradiction {con:.3f} · {expl}</div>
                    </div>
                    """, unsafe_allow_html=True)

            with col_chart:
                # Donut
                label_counts: dict[str, int] = {}
                for v in verdicts:
                    l = v["label"]
                    label_counts[l] = label_counts.get(l, 0) + 1

                colors = {
                    "SUPPORTED": "#16a34a", "CONTRADICTED": "#dc2626",
                    "UNSUPPORTED": "#d97706", "NEUTRAL": "#7c3aed",
                }
                fig = go.Figure(go.Pie(
                    labels=list(label_counts.keys()),
                    values=list(label_counts.values()),
                    hole=0.58,
                    marker=dict(
                        colors=[colors.get(l, "#64748b") for l in label_counts],
                        line=dict(color="#0f172a", width=2),
                    ),
                    textinfo="label+percent",
                    textfont=dict(size=11, color="#e2e8f0"),
                ))
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#94a3b8"), showlegend=False,
                    margin=dict(l=10, r=10, t=10, b=10), height=240,
                    annotations=[dict(
                        text=f"<b>{faithfulness:.0%}</b>", x=0.5, y=0.5,
                        font=dict(size=22, color=faith_color, family="Inter"),
                        showarrow=False,
                    )],
                )
                st.plotly_chart(fig, use_container_width=True)

                # Bar chart
                if len(verdicts) <= 8:
                    short_claims = [v["claim"][:35] + "…" for v in verdicts]
                    fig2 = go.Figure()
                    fig2.add_trace(go.Bar(
                        name="Entailment", x=short_claims,
                        y=[v["entailment_score"] for v in verdicts],
                        marker_color="#16a34a", opacity=0.85,
                    ))
                    fig2.add_trace(go.Bar(
                        name="Contradiction", x=short_claims,
                        y=[v["contradiction_score"] for v in verdicts],
                        marker_color="#dc2626", opacity=0.85,
                    ))
                    fig2.update_layout(
                        barmode="group", paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8", size=9),
                        margin=dict(l=10, r=10, t=10, b=60), height=220,
                        legend=dict(orientation="h", y=-0.4, font=dict(size=9)),
                        xaxis=dict(tickangle=-30, showgrid=False, tickfont=dict(size=8)),
                        yaxis=dict(showgrid=True, gridcolor="#1e293b"),
                    )
                    fig2.add_hline(y=0.65, line_dash="dot", line_color="#34d399",
                                   annotation_text="Support threshold",
                                   annotation_font_color="#34d399",
                                   annotation_font_size=9)
                    st.plotly_chart(fig2, use_container_width=True)

    # ── Tab 3: Sources ────────────────────────────────────────────────────
    with tab_src:
        if not passages:
            st.info("No evidence passages retrieved.")
        else:
            for i, p in enumerate(passages):
                title = p.get("title", "Untitled")
                abstract = p.get("abstract", "")
                source = p.get("source_label", f"PMID: {p.get('pmid','?')}")
                preview = abstract[:400] + ("…" if len(abstract) > 400 else "")
                st.markdown(f"""
                <div class="passage">
                  <div class="passage-title">{i+1}. {title}</div>
                  <div class="passage-body">{preview}</div>
                  <div class="passage-meta">{source}</div>
                </div>
                """, unsafe_allow_html=True)

    # ── Tab 4: Trace ──────────────────────────────────────────────────────
    with tab_trace:
        for entry in trace:
            st.markdown(f'<div class="trace-line">{entry}</div>', unsafe_allow_html=True)

elif run and not query:
    st.warning("Enter a biomedical question first.")
else:
    # Landing state
    st.markdown("")
    c1, c2, c3 = st.columns(3)
    for col, icon, title, desc in [
        (c1, "📚", "BioASQ-style Retrieval", "Searches PubMed for relevant clinical and research literature using query-optimized E-utilities."),
        (c2, "🧪", "DeBERTa NLI Verification", "Every generated claim is scored for entailment, contradiction, and neutrality against retrieved passages."),
        (c3, "🔄", "Self-Correction Loop", "Unsupported claims are automatically removed. The agent abstains rather than hallucinate medical facts."),
    ]:
        with col:
            st.markdown(f"""
            <div class="card" style="text-align:center;padding:1.5rem 1.2rem">
              <div style="font-size:1.8rem;margin-bottom:0.6rem">{icon}</div>
              <div style="font-size:0.88rem;font-weight:700;color:#e2e8f0;margin-bottom:0.5rem">{title}</div>
              <div style="font-size:0.8rem;color:#64748b;line-height:1.6">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;color:#475569;font-size:0.8rem;margin-top:1.5rem">
      Type a biomedical question above and click <b>Analyze →</b>
    </div>
    """, unsafe_allow_html=True)
