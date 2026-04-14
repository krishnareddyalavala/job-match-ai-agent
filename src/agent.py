# src/agent.py
"""
LangGraph-based Job Match Agent.

Nodes (steps):
1. parse_jd      - Extract key requirements from the JD
2. score_match   - Score how well the resume matches (0-100)
3. gap_analysis  - Identify missing/weak skills vs JD
4. rewrite       - Rewrite summary + key bullets aligned to JD
5. report        - Compile final report

State flows through all nodes sequentially.
"""

import os
import json
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()


# ── LLM Setup ──────────────────────────────────────────────────────────────────

def get_llm(temperature: float = 0.3) -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=temperature,
    )


# ── Agent State ────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    # Inputs
    resume_text: str
    jd_text: str
    resume_context: str        # RAG-retrieved relevant chunks

    # Intermediate outputs
    jd_requirements: list      # Parsed JD requirements
    match_score: int           # 0-100
    matched_skills: list       # Skills found in resume
    missing_skills: list       # Skills missing from resume
    weak_areas: list           # Skills mentioned but not detailed

    # Rewritten content
    rewritten_summary: str
    rewritten_bullets: list    # List of improved bullet points

    # Final
    final_report: str
    status: str                # Track progress


# ── Node 1: Parse JD ──────────────────────────────────────────────────────────

def parse_jd_node(state: AgentState) -> AgentState:
    """Extract structured requirements from the job description."""
    llm = get_llm(temperature=0.1)

    prompt = f"""You are an expert technical recruiter. 
Analyze the following Job Description and extract ALL requirements into structured categories.

JOB DESCRIPTION:
{state['jd_text']}

Return a JSON object with exactly these keys:
{{
  "role_title": "exact job title",
  "must_have_skills": ["skill1", "skill2", ...],
  "nice_to_have_skills": ["skill1", "skill2", ...],
  "experience_required": "e.g. 5+ years in ...",
  "key_responsibilities": ["responsibility1", "responsibility2", ...],
  "domain_knowledge": ["domain1", "domain2", ...],
  "tools_and_technologies": ["tool1", "tool2", ...]
}}

Return ONLY valid JSON, no explanation."""

    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        # Clean up response and parse JSON
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        requirements = json.loads(content.strip())
    except Exception:
        requirements = {
            "role_title": "Unknown",
            "must_have_skills": [],
            "nice_to_have_skills": [],
            "experience_required": "",
            "key_responsibilities": [],
            "domain_knowledge": [],
            "tools_and_technologies": []
        }

    return {
        **state,
        "jd_requirements": requirements,
        "status": "✅ JD parsed successfully"
    }


# ── Node 2: Score Match ────────────────────────────────────────────────────────

def score_match_node(state: AgentState) -> AgentState:
    """Score how well the resume matches the JD requirements."""
    llm = get_llm(temperature=0.1)

    requirements = state["jd_requirements"]

    prompt = f"""You are an expert ATS system and technical recruiter.
Score the following resume against the job requirements.

RESUME:
{state['resume_text']}

JOB REQUIREMENTS:
{json.dumps(requirements, indent=2)}

Analyze carefully and return a JSON object:
{{
  "overall_score": <integer 0-100>,
  "matched_skills": ["skill1", "skill2", ...],
  "missing_skills": ["skill1", "skill2", ...],
  "weak_areas": ["area that is mentioned but not detailed enough"],
  "score_breakdown": {{
    "technical_skills": <0-100>,
    "experience_relevance": <0-100>,
    "domain_knowledge": <0-100>,
    "tools_match": <0-100>
  }},
  "strengths": ["strength1", "strength2", ...],
  "improvement_areas": ["area1", "area2", ...]
}}

Be honest and precise. Return ONLY valid JSON."""

    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        result = json.loads(content.strip())
    except Exception:
        result = {
            "overall_score": 0,
            "matched_skills": [],
            "missing_skills": [],
            "weak_areas": [],
            "score_breakdown": {},
            "strengths": [],
            "improvement_areas": []
        }

    return {
        **state,
        "match_score": result.get("overall_score", 0),
        "matched_skills": result.get("matched_skills", []),
        "missing_skills": result.get("missing_skills", []),
        "weak_areas": result.get("weak_areas", []),
        "status": f"✅ Match score: {result.get('overall_score', 0)}/100"
    }


# ── Node 3: Gap Analysis ───────────────────────────────────────────────────────

def gap_analysis_node(state: AgentState) -> AgentState:
    """Deep analysis of gaps between resume and JD."""
    # This node uses the scoring data to prepare context for rewriting
    # It's lightweight - gap data is already in state from score_match_node

    gaps_summary = []
    if state.get("missing_skills"):
        gaps_summary.append(f"MISSING: {', '.join(state['missing_skills'][:10])}")
    if state.get("weak_areas"):
        gaps_summary.append(f"WEAK AREAS: {', '.join(state['weak_areas'][:5])}")

    return {
        **state,
        "status": f"✅ Gap analysis complete — {len(state.get('missing_skills', []))} gaps found"
    }


# ── Node 4: Rewrite ────────────────────────────────────────────────────────────

def rewrite_node(state: AgentState) -> AgentState:
    """Rewrite resume summary and key bullet points to better match the JD."""
    llm = get_llm(temperature=0.4)

    requirements = state["jd_requirements"]
    matched = state.get("matched_skills", [])
    missing = state.get("missing_skills", [])

    # --- Rewrite Summary ---
    summary_prompt = f"""You are an expert resume writer. 
Rewrite the professional summary below to better match the job description requirements.

ORIGINAL SUMMARY (extract from resume):
{state['resume_context'][:2000]}

JOB TITLE: {requirements.get('role_title', '')}
MUST-HAVE SKILLS: {', '.join(requirements.get('must_have_skills', []))}
KEY RESPONSIBILITIES: {'; '.join(requirements.get('key_responsibilities', [])[:5])}

MATCHED SKILLS (already in resume): {', '.join(matched[:10])}
SKILLS TO EMPHASIZE MORE: {', '.join(missing[:5])}

Rules:
- Keep it 3-4 sentences max
- Use keywords from the JD naturally
- Do NOT invent experience that isn't implied by the original
- Keep the same person's voice and experience level
- Make it ATS-friendly

Return ONLY the rewritten summary paragraph, no explanation."""

    summary_response = llm.invoke([HumanMessage(content=summary_prompt)])
    rewritten_summary = summary_response.content.strip()

    # --- Rewrite Key Bullets ---
    bullets_prompt = f"""You are an expert resume writer.
Given the job description requirements, rewrite or enhance the following resume experience bullets to better align with the role.

RESUME CONTEXT (relevant experience):
{state['resume_context']}

JOB REQUIREMENTS:
- Role: {requirements.get('role_title', '')}
- Must-have skills: {', '.join(requirements.get('must_have_skills', []))}
- Responsibilities: {'; '.join(requirements.get('key_responsibilities', [])[:6])}

Rules:
- Write 6-8 strong bullet points
- Start each with a strong action verb
- Quantify where possible (use realistic numbers based on context)
- Naturally weave in JD keywords
- Do NOT fabricate tools or experience not present in the original
- Format: return a JSON array of strings like ["bullet1", "bullet2", ...]

Return ONLY a valid JSON array of bullet strings."""

    bullets_response = llm.invoke([HumanMessage(content=bullets_prompt)])

    try:
        content = bullets_response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        rewritten_bullets = json.loads(content.strip())
    except Exception:
        rewritten_bullets = ["Could not generate bullets. Please retry."]

    return {
        **state,
        "rewritten_summary": rewritten_summary,
        "rewritten_bullets": rewritten_bullets,
        "status": "✅ Resume sections rewritten"
    }


# ── Node 5: Final Report ───────────────────────────────────────────────────────

def report_node(state: AgentState) -> AgentState:
    """Compile everything into a final readable report."""

    requirements = state.get("jd_requirements", {})
    score = state.get("match_score", 0)

    # Score color label
    if score >= 80:
        score_label = "🟢 STRONG MATCH"
    elif score >= 60:
        score_label = "🟡 MODERATE MATCH"
    else:
        score_label = "🔴 WEAK MATCH"

    matched = state.get("matched_skills", [])
    missing = state.get("missing_skills", [])
    weak = state.get("weak_areas", [])

    bullets_text = "\n".join([f"• {b}" for b in state.get("rewritten_bullets", [])])

    report = f"""
╔══════════════════════════════════════════════════════╗
   JOB MATCH REPORT — {requirements.get('role_title', 'Unknown Role')}
╚══════════════════════════════════════════════════════╝

MATCH SCORE: {score}/100  {score_label}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ MATCHED SKILLS ({len(matched)}):
{chr(10).join([f"  • {s}" for s in matched]) or "  None identified"}

❌ MISSING / GAPS ({len(missing)}):
{chr(10).join([f"  • {s}" for s in missing]) or "  None — great match!"}

⚠️  WEAK AREAS (mentioned but not detailed):
{chr(10).join([f"  • {s}" for s in weak]) or "  None"}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📝 REWRITTEN PROFESSIONAL SUMMARY:
{state.get('rewritten_summary', '')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💼 REWRITTEN EXPERIENCE BULLETS (aligned to JD):
{bullets_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 NEXT STEPS:
  1. Add these bullets to your resume under the most relevant role
  2. Replace your current summary with the rewritten version
  3. Learn / highlight: {', '.join(missing[:3]) if missing else 'You are well-aligned!'}
  4. Tailor your LinkedIn headline to include: {requirements.get('role_title', '')}
"""

    return {
        **state,
        "final_report": report,
        "status": "✅ Report complete!"
    }


# ── Build the Graph ────────────────────────────────────────────────────────────

def build_agent() -> StateGraph:
    """Build and compile the LangGraph agent."""

    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("parse_jd", parse_jd_node)
    graph.add_node("score_match", score_match_node)
    graph.add_node("gap_analysis", gap_analysis_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("report", report_node)

    # Define edges (sequential flow)
    graph.set_entry_point("parse_jd")
    graph.add_edge("parse_jd", "score_match")
    graph.add_edge("score_match", "gap_analysis")
    graph.add_edge("gap_analysis", "rewrite")
    graph.add_edge("rewrite", "report")
    graph.add_edge("report", END)

    return graph.compile()


# ── Run Agent ──────────────────────────────────────────────────────────────────

def run_agent(resume_text: str, jd_text: str, resume_context: str) -> AgentState:
    """Run the full agent pipeline and return final state."""
    agent = build_agent()

    initial_state: AgentState = {
        "resume_text": resume_text,
        "jd_text": jd_text,
        "resume_context": resume_context,
        "jd_requirements": [],
        "match_score": 0,
        "matched_skills": [],
        "missing_skills": [],
        "weak_areas": [],
        "rewritten_summary": "",
        "rewritten_bullets": [],
        "final_report": "",
        "status": "Starting..."
    }

    final_state = agent.invoke(initial_state)
    return final_state
