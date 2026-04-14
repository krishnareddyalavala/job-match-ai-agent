# app.py
"""
Job Match AI Agent — Streamlit Frontend
Paste your JD, upload your resume, and get:
  - Match score
  - Gap analysis
  - Rewritten summary & bullets
  - Downloadable .docx
"""

import os
import sys
import tempfile
import streamlit as st
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from resume_parser import parse_resume, chunk_resume
from rag_engine import ResumeRAG
from agent import run_agent
from exporter import export_to_docx, export_report_to_txt

load_dotenv()

# ── Page Config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Job Match AI Agent",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1F497D, #2E75B6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .score-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 2rem;
        font-weight: 800;
    }
    .score-green { background: #d4edda; color: #155724; }
    .score-yellow { background: #fff3cd; color: #856404; }
    .score-red { background: #f8d7da; color: #721c24; }
    .skill-chip {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        margin: 3px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .chip-green { background: #d4edda; color: #155724; }
    .chip-red { background: #f8d7da; color: #721c24; }
    .chip-yellow { background: #fff3cd; color: #856404; }
    .step-box {
        background: #f8f9fa;
        border-left: 4px solid #2E75B6;
        padding: 10px 15px;
        margin: 8px 0;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
    st.markdown("## ⚙️ Settings")

    st.markdown("**OpenAI Config**")
    api_key = st.text_input("OpenAI API Key (sk-...)", value=os.getenv("OPENAI_API_KEY", ""), type="password")
    model = st.text_input("Model", value=os.getenv("OPENAI_MODEL", "gpt-4o"))

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if model:
        os.environ["OPENAI_MODEL"] = model

    st.divider()
    st.markdown("**How it works:**")
    st.markdown("""
    1. 📄 Upload your resume
    2. 📋 Paste the Job Description
    3. 🤖 Agent analyzes & scores
    4. ✍️ Get rewritten sections
    5. 📥 Download updated resume
    """)

    st.divider()
    st.markdown("**Built with:**")
    st.markdown("LangGraph • Azure OpenAI • FAISS • RAG • Streamlit")


# ── Main Header ────────────────────────────────────────────────────────────────

st.markdown('<p class="main-header">🎯 Job Match AI Agent</p>', unsafe_allow_html=True)
st.markdown("*Upload your resume + paste a JD → Get match score, gap analysis & rewritten resume sections*")
st.divider()


# ── Input Section ──────────────────────────────────────────────────────────────

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📄 Your Resume")
    uploaded_file = st.file_uploader(
        "Upload resume (.docx or .pdf)",
        type=["docx", "pdf"],
        help="Upload your current resume"
    )
    if uploaded_file:
        st.success(f"✅ Loaded: {uploaded_file.name}")

with col2:
    st.markdown("### 📋 Job Description")
    jd_text = st.text_area(
        "Paste the full Job Description here",
        height=300,
        placeholder="Paste the complete job description including responsibilities, requirements, and skills..."
    )
    if jd_text:
        st.caption(f"📝 {len(jd_text.split())} words | {len(jd_text)} characters")


# ── Run Button ─────────────────────────────────────────────────────────────────

st.divider()

run_col, _, _ = st.columns([1, 2, 2])
with run_col:
    run_clicked = st.button(
        "🚀 Analyze & Match",
        type="primary",
        use_container_width=True,
        disabled=not (uploaded_file and jd_text and api_key)
    )

if not api_key:
    st.warning("⚠️ Please enter your OpenAI API Key (sk-...) in the sidebar to get started.")


# ── Agent Pipeline ─────────────────────────────────────────────────────────────

if run_clicked and uploaded_file and jd_text:

    # Save uploaded file to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.divider()
    st.markdown("### 🤖 Agent Pipeline Running...")

    progress_bar = st.progress(0)
    status_container = st.empty()

    try:
        # Step 1: Parse Resume
        status_container.markdown('<div class="step-box">📖 Step 1/5 — Parsing resume...</div>', unsafe_allow_html=True)
        progress_bar.progress(10)
        resume_text = parse_resume(tmp_path)
        resume_chunks = chunk_resume(resume_text)

        # Step 2: Build RAG Index
        status_container.markdown('<div class="step-box">🔍 Step 2/5 — Building RAG index from resume...</div>', unsafe_allow_html=True)
        progress_bar.progress(25)
        rag = ResumeRAG()
        rag.build_index(resume_chunks)

        # Retrieve relevant context using JD as query
        resume_context = rag.get_full_context(
            queries=[jd_text[:500], jd_text[500:1000] if len(jd_text) > 500 else jd_text],
            top_k=4
        )

        # Step 3: Run LangGraph Agent
        status_container.markdown('<div class="step-box">🧠 Step 3/5 — Running LangGraph agent (parse JD → score → gap analysis → rewrite → report)...</div>', unsafe_allow_html=True)
        progress_bar.progress(40)

        final_state = run_agent(
            resume_text=resume_text,
            jd_text=jd_text,
            resume_context=resume_context
        )

        progress_bar.progress(85)

        # Step 4: Export
        status_container.markdown('<div class="step-box">💾 Step 4/5 — Exporting updated resume...</div>', unsafe_allow_html=True)
        role_title = final_state.get("jd_requirements", {}).get("role_title", "Role")
        docx_path = export_to_docx(
            original_resume_path=tmp_path,
            rewritten_summary=final_state.get("rewritten_summary", ""),
            rewritten_bullets=final_state.get("rewritten_bullets", []),
            job_title=role_title,
            output_dir="outputs"
        )
        report_path = export_report_to_txt(
            final_state.get("final_report", ""),
            output_dir="outputs"
        )

        progress_bar.progress(100)
        status_container.markdown('<div class="step-box">✅ Step 5/5 — Complete!</div>', unsafe_allow_html=True)

        # ── Results ───────────────────────────────────────────────────────────

        st.divider()
        st.markdown("## 📊 Results")

        # Score Card
        score = final_state.get("match_score", 0)
        if score >= 80:
            score_class = "score-green"
            score_emoji = "🟢"
            score_label = "STRONG MATCH"
        elif score >= 60:
            score_class = "score-yellow"
            score_emoji = "🟡"
            score_label = "MODERATE MATCH"
        else:
            score_class = "score-red"
            score_emoji = "🔴"
            score_label = "NEEDS WORK"

        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.markdown(f"""
            <div class="score-box {score_class}">
                {score_emoji} {score}/100<br>
                <span style="font-size:1rem">{score_label}</span>
            </div>
            """, unsafe_allow_html=True)
        with sc2:
            st.metric("✅ Matched Skills", len(final_state.get("matched_skills", [])))
        with sc3:
            st.metric("❌ Missing Skills", len(final_state.get("missing_skills", [])))

        st.divider()

        # Skills breakdown
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Skills Analysis", "✍️ Rewritten Content", "📋 Full Report", "📥 Downloads"])

        with tab1:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("#### ✅ Matched Skills")
                matched = final_state.get("matched_skills", [])
                chips = " ".join([f'<span class="skill-chip chip-green">{s}</span>' for s in matched])
                st.markdown(chips or "None found", unsafe_allow_html=True)

                st.markdown("#### ⚠️ Weak Areas")
                weak = final_state.get("weak_areas", [])
                w_chips = " ".join([f'<span class="skill-chip chip-yellow">{s}</span>' for s in weak])
                st.markdown(w_chips or "None", unsafe_allow_html=True)

            with col_b:
                st.markdown("#### ❌ Missing Skills / Gaps")
                missing = final_state.get("missing_skills", [])
                m_chips = " ".join([f'<span class="skill-chip chip-red">{s}</span>' for s in missing])
                st.markdown(m_chips or "Great — no major gaps!", unsafe_allow_html=True)

                st.markdown("#### 📌 JD Requirements Extracted")
                reqs = final_state.get("jd_requirements", {})
                if reqs:
                    st.markdown(f"**Role:** {reqs.get('role_title', '')}")
                    st.markdown(f"**Experience:** {reqs.get('experience_required', '')}")
                    must = reqs.get("must_have_skills", [])
                    if must:
                        st.markdown("**Must-have:**")
                        for s in must:
                            st.markdown(f"  - {s}")

        with tab2:
            st.markdown("#### 📝 Rewritten Professional Summary")
            st.info(final_state.get("rewritten_summary", "Not generated"))

            st.markdown("#### 💼 Rewritten Experience Bullets")
            bullets = final_state.get("rewritten_bullets", [])
            for i, bullet in enumerate(bullets, 1):
                st.markdown(f"**{i}.** {bullet}")

            st.caption("💡 Copy these into your resume under your most relevant role")

        with tab3:
            st.markdown("#### 📋 Full Match Report")
            st.code(final_state.get("final_report", ""), language=None)

        with tab4:
            st.markdown("#### 📥 Download Your Files")

            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                if os.path.exists(docx_path):
                    with open(docx_path, "rb") as f:
                        st.download_button(
                            label="📄 Download Aligned Resume (.docx)",
                            data=f.read(),
                            file_name=os.path.basename(docx_path),
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            use_container_width=True
                        )
            with dl_col2:
                if os.path.exists(report_path):
                    with open(report_path, "r", encoding="utf-8") as f:
                        st.download_button(
                            label="📋 Download Match Report (.txt)",
                            data=f.read(),
                            file_name=os.path.basename(report_path),
                            mime="text/plain",
                            use_container_width=True
                        )

            st.caption("The .docx file contains your original resume + a new 'JD-Aligned Version' section at the end.")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ── Footer ─────────────────────────────────────────────────────────────────────

st.divider()
st.markdown(
    "<div style='text-align:center; color:#888; font-size:0.8rem'>"
    "Built with LangGraph • Azure OpenAI • FAISS RAG • Streamlit | Krishna Reddy Alavala"
    "</div>",
    unsafe_allow_html=True
)
