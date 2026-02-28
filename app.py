"""
app.py
------
AI Resume–Job Description Matcher & Career Advisor
Main Streamlit application entry point.

Run with:
    streamlit run app.py
"""

import logging
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Load .env file before anything else so OPENAI_API_KEY is available
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# Ensure src/ is importable when running from project root
sys.path.insert(0, str(Path(__file__).parent))

from src.pdf_parser import parse_pdf
from src.nlp_pipeline import NLPPipeline
from src.scoring import compute_match_score, score_label
from src.llm_advisor import generate_career_advice

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Resume Matcher",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — professional dark-accent theme
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ---- Global font & background ---- */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    /* ---- Header gradient ---- */
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        color: white;
    }
    .main-header p {
        font-size: 1rem;
        color: #94a3b8;
        margin-top: 0.4rem;
    }

    /* ---- Score gauge card ---- */
    .score-card {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        border: 1px solid #334155;
    }
    .score-number {
        font-size: 5rem;
        font-weight: 800;
        line-height: 1;
    }
    .score-label {
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }

    /* ---- Metric cards ---- */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: #1e293b;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        flex: 1;
        border-left: 4px solid #3b82f6;
    }
    .metric-card .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #60a5fa;
    }
    .metric-card .metric-name {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* ---- Skill pills ---- */
    .skill-pill {
        display: inline-block;
        padding: 0.2rem 0.75rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 500;
        margin: 0.2rem;
    }
    .pill-match  { background: #166534; color: #bbf7d0; border: 1px solid #22c55e; }
    .pill-missing { background: #7f1d1d; color: #fecaca; border: 1px solid #ef4444; }
    .pill-extra  { background: #1e3a5f; color: #bfdbfe; border: 1px solid #3b82f6; }

    /* ---- Section headers ---- */
    .section-header {
        font-size: 1.15rem;
        font-weight: 600;
        color: #e2e8f0;
        border-bottom: 2px solid #334155;
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }

    /* ---- Progress bar custom ---- */
    .stProgress > div > div > div > div {
        background-color: #3b82f6;
    }

    /* ---- Advice box ---- */
    .advice-container {
        background: #1e293b;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        border: 1px solid #334155;
    }

    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] {
        background: #0f172a;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.divider()

    st.markdown("### 🤖 AI Career Advisor")

    # --- Groq API key ---
    _groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if _groq_key:
        st.success("✅ Groq API key loaded from .env")
    else:
        st.warning("Groq key not found in .env")

    groq_key_input = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="Loaded from .env" if _groq_key else "gsk_...",
        help="Pre-filled from your .env file. Paste here to override.",
    )
    resolved_groq_key = groq_key_input.strip() if groq_key_input.strip() else _groq_key

    groq_model = st.selectbox(
        "Groq Model",
        options=[
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "meta-llama/llama-4-maverick-17b-128e-instruct",
            "llama-3.3-70b-versatile",
            "llama3-70b-8192",
        ],
        index=0,
        help="Groq model used to generate career insights.",
    )

    st.divider()
    st.markdown("### 📊 Scoring Weights")
    st.caption("Current formula:")
    st.code(
        "Score = 0.60 × DocSim\n"
        "      + 0.30 × SkillOverlap\n"
        "      + 0.10 × ExactBonus",
        language="text",
    )

    st.divider()
    st.markdown("### ℹ️ About")
    st.caption(
        "AI Resume Matcher uses Sentence-BERT for semantic similarity "
        "and a curated skill taxonomy for skill gap analysis."
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>🎯 AI Resume–Job Description Matcher</h1>
        <p>Semantic NLP-powered career analysis & personalized learning roadmap</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading NLP models (first run only)...")
def load_pipeline() -> NLPPipeline:
    """Load and cache the NLP pipeline."""
    return NLPPipeline()


# ---------------------------------------------------------------------------
# Helper: render skill pills
# ---------------------------------------------------------------------------
def render_skill_pills(skills: list[str], pill_class: str) -> str:
    """Render a list of skills as HTML pill badges."""
    if not skills:
        return "<em style='color:#64748b'>None detected</em>"
    pills = "".join(
        f'<span class="skill-pill {pill_class}">{s.replace("<","&lt;")}</span>'
        for s in skills
    )
    return pills


# ---------------------------------------------------------------------------
# Helper: gauge SVG
# ---------------------------------------------------------------------------
def render_gauge(score: float, color: str) -> str:
    """
    Render an SVG arc-style gauge for the match score.

    Args:
        score: Score in [0, 100].
        color: Hex color string for the arc.

    Returns:
        HTML string with embedded SVG.
    """
    import math

    radius = 90
    cx, cy = 110, 110
    stroke_width = 18
    circumference = math.pi * radius  # half circle
    progress = (score / 100) * circumference

    # Arc path for background (180°)
    bg_d = f"M {cx - radius},{cy} A {radius},{radius} 0 0,1 {cx + radius},{cy}"
    fg_d = bg_d  # same path, different dash

    svg = f"""
    <svg viewBox="0 0 220 130" xmlns="http://www.w3.org/2000/svg" width="260">
      <!-- Background arc -->
      <path d="{bg_d}" fill="none" stroke="#1e293b" stroke-width="{stroke_width}"
            stroke-linecap="round"/>
      <!-- Foreground arc (progress) -->
      <path d="{fg_d}" fill="none" stroke="{color}" stroke-width="{stroke_width}"
            stroke-linecap="round"
            stroke-dasharray="{progress:.1f} {circumference:.1f}"
            stroke-dashoffset="0"/>
      <!-- Score text -->
      <text x="{cx}" y="{cy + 10}" text-anchor="middle"
            font-size="32" font-weight="800" fill="{color}" font-family="Inter,sans-serif">
        {score:.0f}%
      </text>
      <text x="{cx}" y="{cy + 32}" text-anchor="middle"
            font-size="11" fill="#94a3b8" font-family="Inter,sans-serif">
        MATCH SCORE
      </text>
    </svg>
    """
    return svg


# ---------------------------------------------------------------------------
# Main UI — Upload Section
# ---------------------------------------------------------------------------
st.markdown("### 📄 Upload Documents")
col_resume, col_jd = st.columns(2)

with col_resume:
    resume_file = st.file_uploader(
        "Upload Resume (PDF)",
        type=["pdf"],
        help="Upload your resume as a PDF file.",
        key="resume_upload",
    )
    if resume_file:
        st.success(f"Loaded: **{resume_file.name}**")

with col_jd:
    jd_file = st.file_uploader(
        "Upload Job Description (PDF)",
        type=["pdf"],
        help="Upload the job description as a PDF file.",
        key="jd_upload",
    )
    if jd_file:
        st.success(f"Loaded: **{jd_file.name}**")

st.divider()

# ---------------------------------------------------------------------------
# Analysis trigger
# ---------------------------------------------------------------------------
run_button = st.button(
    "🚀 Analyze Match",
    type="primary",
    use_container_width=True,
    disabled=(resume_file is None or jd_file is None),
)

if resume_file is None or jd_file is None:
    st.info("Upload both a resume and a job description PDF to begin analysis.")
    st.stop()


# ---------------------------------------------------------------------------
# Run analysis
# ---------------------------------------------------------------------------
if run_button or ("analysis_result" in st.session_state):

    if run_button:
        # -- Parse PDFs --
        with st.spinner("Extracting text from PDFs..."):
            try:
                resume_data = parse_pdf(resume_file.read())
                jd_data = parse_pdf(jd_file.read())
            except ValueError as exc:
                st.error(f"PDF parsing error: {exc}")
                st.stop()

        # -- Run NLP pipeline --
        pipeline = load_pipeline()
        with st.spinner("Running semantic NLP analysis..."):
            nlp_result = pipeline.analyze(
                resume_text=resume_data["clean_text"],
                jd_text=jd_data["clean_text"],
            )

        # -- Compute match score --
        with st.spinner("Computing match score..."):
            match_result = compute_match_score(
                doc_similarity=nlp_result["doc_similarity"],
                skill_similarity=nlp_result["skill_similarity"],
                resume_skills=nlp_result["resume_skills"],
                jd_skills=nlp_result["jd_skills"],
            )

        # -- Generate career advice --
        with st.spinner("Generating AI career advice (Groq)..."):
            advice_result = generate_career_advice(
                resume_skills=nlp_result["resume_skills"],
                jd_skills=nlp_result["jd_skills"],
                matching_skills=match_result.matching_skills,
                missing_skills=match_result.missing_skills,
                extra_skills=match_result.extra_skills,
                match_score=match_result.match_score,
                doc_similarity=match_result.doc_similarity,
                groq_api_key=resolved_groq_key or None,
                groq_model=groq_model,
            )

        # Cache in session state
        st.session_state["analysis_result"] = {
            "resume_data": resume_data,
            "jd_data": jd_data,
            "nlp_result": nlp_result,
            "match_result": match_result,
            "advice_result": advice_result,
        }

    # Retrieve from session
    result = st.session_state["analysis_result"]
    resume_data = result["resume_data"]
    jd_data = result["jd_data"]
    nlp_result = result["nlp_result"]
    match_result = result["match_result"]
    advice_result = result["advice_result"]

    # -----------------------------------------------------------------------
    # Results Header banner
    # -----------------------------------------------------------------------
    label, color = score_label(match_result.match_score)
    st.success(f"Analysis complete! — {label}")

    # -----------------------------------------------------------------------
    # Row 1: Score gauge + key metrics
    # -----------------------------------------------------------------------
    st.markdown("## 📊 Match Analysis")
    col_gauge, col_metrics = st.columns([1, 2])

    with col_gauge:
        gauge_html = render_gauge(match_result.match_score, color)
        st.markdown(
            f"""
            <div class="score-card">
                {gauge_html}
                <div class="score-label" style="color:{color};">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_metrics:
        m1, m2 = st.columns(2)
        m3, m4 = st.columns(2)

        m1.metric(
            "Document Similarity",
            f"{match_result.doc_similarity:.1f}%",
            help="Semantic similarity of the full resume vs. full JD text.",
        )
        m2.metric(
            "Skill Similarity",
            f"{match_result.skill_similarity:.1f}%",
            help="Semantic similarity of extracted skill sets.",
        )
        m3.metric(
            "Skill Overlap Ratio",
            f"{match_result.skill_overlap_ratio:.1f}%",
            help="% of JD skills found in resume.",
        )
        m4.metric(
            "Skills Matched",
            f"{len(match_result.matching_skills)} / {len(nlp_result['jd_skills'])}",
            help="Number of JD skills matched in resume.",
        )

        # Score breakdown bar chart
        st.markdown("**Score Breakdown**")
        breakdown_df = pd.DataFrame(
            {
                "Component": ["Doc Similarity (60%)", "Skill Overlap (30%)", "Exact Bonus (10%)"],
                "Contribution": [
                    match_result.component_scores["doc_similarity_contribution"],
                    match_result.component_scores["skill_overlap_contribution"],
                    match_result.component_scores["exact_bonus_contribution"],
                ],
            }
        )
        st.bar_chart(breakdown_df.set_index("Component"), height=180)

    st.divider()

    # -----------------------------------------------------------------------
    # Row 2: Skill Analysis
    # -----------------------------------------------------------------------
    st.markdown("## 🔍 Skill Gap Analysis")

    tab_matching, tab_missing, tab_extra, tab_table = st.tabs([
        f"✅ Matching ({len(match_result.matching_skills)})",
        f"❌ Missing ({len(match_result.missing_skills)})",
        f"➕ Extra ({len(match_result.extra_skills)})",
        "📋 Full Comparison Table",
    ])

    with tab_matching:
        st.markdown('<div class="section-header">Skills in Your Resume ∩ Job Description</div>', unsafe_allow_html=True)
        if match_result.matching_skills:
            st.markdown(
                render_skill_pills(match_result.matching_skills, "pill-match"),
                unsafe_allow_html=True,
            )
        else:
            st.warning("No direct skill matches found.")

    with tab_missing:
        st.markdown('<div class="section-header">Skills Required by JD — Not in Your Resume</div>', unsafe_allow_html=True)
        if match_result.missing_skills:
            st.markdown(
                render_skill_pills(match_result.missing_skills, "pill-missing"),
                unsafe_allow_html=True,
            )
            st.caption(f"You are missing **{len(match_result.missing_skills)}** required skills. Focus your upskilling here.")
        else:
            st.success("No critical skill gaps detected!")

    with tab_extra:
        st.markdown('<div class="section-header">Additional Skills in Your Resume (Beyond JD)</div>', unsafe_allow_html=True)
        if match_result.extra_skills:
            st.markdown(
                render_skill_pills(match_result.extra_skills, "pill-extra"),
                unsafe_allow_html=True,
            )
            st.caption("These skills differentiate you and may be valuable for the role.")
        else:
            st.info("No extra skills detected beyond the JD requirements.")

    with tab_table:
        st.markdown('<div class="section-header">Complete Skill Comparison</div>', unsafe_allow_html=True)

        all_skills = sorted(
            set(match_result.matching_skills)
            | set(match_result.missing_skills)
            | set(match_result.extra_skills)
        )

        resume_set = set(match_result.matching_skills) | set(match_result.extra_skills)
        jd_set = set(match_result.matching_skills) | set(match_result.missing_skills)

        table_data = []
        for skill in all_skills:
            status = []
            if skill in match_result.matching_skills:
                status_str = "✅ Match"
            elif skill in match_result.missing_skills:
                status_str = "❌ Missing"
            else:
                status_str = "➕ Extra"
            table_data.append({
                "Skill": skill.title(),
                "In Resume": "✅" if skill in resume_set else "❌",
                "In JD": "✅" if skill in jd_set else "❌",
                "Status": status_str,
            })

        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, hide_index=True, height=400)

    st.divider()

    # -----------------------------------------------------------------------
    # Row 3: Skill Categories breakdown
    # -----------------------------------------------------------------------
    with st.expander("📂 Skill Categories Breakdown", expanded=False):
        cat_col1, cat_col2 = st.columns(2)

        with cat_col1:
            st.markdown("**Resume Skill Categories**")
            resume_cats = nlp_result.get("resume_skill_categories", {})
            if resume_cats:
                for cat, skills in resume_cats.items():
                    st.markdown(f"**{cat.replace('_', ' ').title()}**: {', '.join(s.title() for s in skills)}")
            else:
                st.write("No categorized skills found.")

        with cat_col2:
            st.markdown("**JD Skill Categories**")
            jd_cats = nlp_result.get("jd_skill_categories", {})
            if jd_cats:
                for cat, skills in jd_cats.items():
                    st.markdown(f"**{cat.replace('_', ' ').title()}**: {', '.join(s.title() for s in skills)}")
            else:
                st.write("No categorized skills found.")

    # -----------------------------------------------------------------------
    # Row 4: AI Career Advisor
    # -----------------------------------------------------------------------
    st.markdown("## 🤖 AI Career Advisor")

    advisor_source = advice_result.get("source", "rule-based")
    if advisor_source == "groq":
        st.success("⚡ Powered by Groq (Llama 4) — AI-generated career insights based on your match score.")
    else:
        st.info(
            "Using rule-based advisor. "
            "Add your Groq API key in the sidebar (or .env) for AI-generated personalized guidance."
        )

    advice_text = advice_result.get("advice", "No advice generated.")

    # Render advice in an expandable card
    with st.container():
        st.markdown(
            f'<div class="advice-container">{""}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(advice_text)

    st.divider()

    # -----------------------------------------------------------------------
    # Row 5: Extracted Text Preview
    # -----------------------------------------------------------------------
    with st.expander("📝 Extracted Text Preview", expanded=False):
        t1, t2 = st.tabs(["Resume Text", "Job Description Text"])
        with t1:
            st.text_area(
                "Resume (first 3000 chars)",
                value=resume_data["clean_text"][:3000],
                height=300,
                disabled=True,
            )
            st.caption(f"Pages: {resume_data['page_count']} | Chars: {len(resume_data['clean_text']):,}")
        with t2:
            st.text_area(
                "Job Description (first 3000 chars)",
                value=jd_data["clean_text"][:3000],
                height=300,
                disabled=True,
            )
            st.caption(f"Pages: {jd_data['page_count']} | Chars: {len(jd_data['clean_text']):,}")

    # -----------------------------------------------------------------------
    # Row 6: Export results
    # -----------------------------------------------------------------------
    st.markdown("## 💾 Export Results")
    export_col1, export_col2 = st.columns(2)

    with export_col1:
        # Export as CSV
        export_df = pd.DataFrame({
            "Metric": [
                "Match Score (%)",
                "Document Similarity (%)",
                "Skill Similarity (%)",
                "Skill Overlap (%)",
                "Matching Skills",
                "Missing Skills",
                "Extra Skills",
            ],
            "Value": [
                match_result.match_score,
                match_result.doc_similarity,
                match_result.skill_similarity,
                match_result.skill_overlap_ratio,
                ", ".join(match_result.matching_skills),
                ", ".join(match_result.missing_skills),
                ", ".join(match_result.extra_skills),
            ],
        })
        csv_bytes = export_df.to_csv(index=False).encode()
        st.download_button(
            label="⬇️ Download Results (CSV)",
            data=csv_bytes,
            file_name="resume_match_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with export_col2:
        # Export advice as text
        advice_bytes = advice_text.encode("utf-8")
        st.download_button(
            label="⬇️ Download Career Advice (TXT)",
            data=advice_bytes,
            file_name="career_advice.txt",
            mime="text/plain",
            use_container_width=True,
        )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="text-align:center; color:#475569; font-size:0.8rem; margin-top:3rem;">
        AI Resume–Job Description Matcher & Career Advisor
        &nbsp;|&nbsp; Built with Streamlit, spaCy, Sentence-BERT
    </div>
    """,
    unsafe_allow_html=True,
)
