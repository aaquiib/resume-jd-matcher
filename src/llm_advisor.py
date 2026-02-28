"""
llm_advisor.py
--------------
Integrates with the Groq API (Llama 4) to generate professional career
advice based on the structured match analysis.

If no GROQ_API_KEY is configured, the module falls back to a structured
rule-based advisor that produces meaningful (non-AI) career guidance.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_prompt(
    resume_skills: list[str],
    jd_skills: list[str],
    matching_skills: list[str],
    missing_skills: list[str],
    extra_skills: list[str],
    match_score: float,
    doc_similarity: float,
) -> str:
    """
    Construct a detailed prompt for the LLM career advisor.

    Args:
        resume_skills: All skills found in the resume.
        jd_skills: All skills required by the JD.
        matching_skills: Skills present in both.
        missing_skills: JD skills absent from resume.
        extra_skills: Resume skills not in JD.
        match_score: Final weighted match percentage.
        doc_similarity: Document cosine similarity percentage.

    Returns:
        Formatted prompt string.
    """
    prompt = f"""You are a senior career advisor and talent specialist. Analyze the following resume-to-job-description matching data and provide structured, professional career guidance.

## Matching Analysis Data

**Overall Match Score:** {match_score}%
**Document Semantic Similarity:** {doc_similarity}%

**Resume Skills ({len(resume_skills)} total):**
{", ".join(resume_skills) if resume_skills else "None detected"}

**Job Description Required Skills ({len(jd_skills)} total):**
{", ".join(jd_skills) if jd_skills else "None detected"}

**Matching Skills ({len(matching_skills)}):**
{", ".join(matching_skills) if matching_skills else "None"}

**Missing Skills - Gaps ({len(missing_skills)}):**
{", ".join(missing_skills) if missing_skills else "None"}

**Extra Candidate Skills ({len(extra_skills)}):**
{", ".join(extra_skills) if extra_skills else "None"}

---

Based on the data above, provide a comprehensive career advisory report with the following sections:

### 1. Career Strengths
Highlight 3-5 specific strengths this candidate brings to this role based on matching skills and overall similarity.

### 2. Skill Gaps & Weaknesses
Identify the most critical missing skills. Explain WHY each gap matters for this specific role and its potential impact on the candidate's application.

### 3. Job Readiness Assessment
Give an honest, detailed assessment of the candidate's readiness for this role. Be specific about what makes them a strong or weak candidate. Mention the match score's meaning in practical terms.

### 4. Personalized Learning Roadmap
For the top 5 most important missing skills, provide:
- Skill name
- Why it's critical for this role
- Recommended learning resource (specific course, book, or platform)
- Estimated proficiency timeline

### 5. Suggested Projects
Recommend 3 hands-on projects the candidate should build to demonstrate their skills for this specific role. Each project should address one or more skill gaps.

### 6. Application Strategy
Provide 3-5 actionable tips on how the candidate should position themselves when applying for this role given their current skill set.

Keep the tone professional, encouraging yet honest. Be specific and actionable — avoid generic advice.
"""
    return prompt


# ---------------------------------------------------------------------------
# Groq-backed advisor
# ---------------------------------------------------------------------------

def _call_groq_api(
    prompt: str,
    api_key: str,
    model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
) -> str:
    """
    Call the Groq Chat Completions API with streaming.

    Args:
        prompt: Full prompt string.
        api_key: Groq API key.
        model: Model identifier (default: meta-llama/llama-4-scout-17b-16e-instruct).

    Returns:
        LLM response text (streamed and concatenated).

    Raises:
        RuntimeError: On API failure.
    """
    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert career advisor with deep knowledge of "
                        "the tech industry, hiring processes, and professional development. "
                        "Provide structured, actionable, and honest career guidance."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=1,
            max_completion_tokens=2048,
            top_p=1,
            stream=True,
            stop=None,
        )

        # Collect streamed chunks into a single string
        response_text = ""
        for chunk in completion:
            delta = chunk.choices[0].delta.content
            if delta:
                response_text += delta

        return response_text.strip()

    except ImportError:
        raise RuntimeError("groq package is not installed. Run: pip install groq")
    except Exception as exc:
        raise RuntimeError(f"Groq API call failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Rule-based fallback advisor
# ---------------------------------------------------------------------------

def _rule_based_advice(
    resume_skills: list[str],
    jd_skills: list[str],
    matching_skills: list[str],
    missing_skills: list[str],
    extra_skills: list[str],
    match_score: float,
) -> str:
    """
    Generate structured career advice without an LLM (rule-based fallback).

    Args:
        resume_skills: All skills from resume.
        jd_skills: All skills from JD.
        matching_skills: Shared skills.
        missing_skills: Skills to acquire.
        extra_skills: Bonus candidate skills.
        match_score: Final match percentage.

    Returns:
        Formatted markdown advice string.
    """
    # Readiness label
    if match_score >= 80:
        readiness = "**High Readiness** — You are a strong candidate for this role."
    elif match_score >= 65:
        readiness = "**Good Readiness** — You are a competitive candidate with some gaps to address."
    elif match_score >= 50:
        readiness = "**Moderate Readiness** — You have a foundation but significant upskilling is needed."
    else:
        readiness = "**Lower Readiness** — Consider this a stretch goal role requiring substantial preparation."

    strengths_section = ""
    if matching_skills:
        top_strengths = matching_skills[:6]
        strengths_section = (
            "### 1. Career Strengths\n"
            f"Your resume demonstrates alignment in these key areas:\n"
            + "\n".join(f"- **{s.title()}**: Present and aligned with role requirements." for s in top_strengths)
        )
        if extra_skills:
            extra_top = extra_skills[:4]
            strengths_section += (
                f"\n\nAdditionally, your skills in **{', '.join(extra_top)}** "
                "may differentiate you as a candidate with broader expertise."
            )
    else:
        strengths_section = (
            "### 1. Career Strengths\n"
            "No direct skill matches were detected. Focus on highlighting transferable skills "
            "and contextual experience in your application materials."
        )

    gaps_section = "### 2. Skill Gaps & Weaknesses\n"
    if missing_skills:
        gaps_section += (
            f"You are missing **{len(missing_skills)}** skills required by this role:\n"
            + "\n".join(f"- **{s.title()}**: Critical for role performance — prioritize learning." for s in missing_skills[:8])
        )
        if len(missing_skills) > 8:
            gaps_section += f"\n- *...and {len(missing_skills) - 8} more skills.*"
    else:
        gaps_section += "No critical skill gaps detected — your profile covers the JD requirements well."

    roadmap_section = "### 3. Personalized Learning Roadmap\n"
    if missing_skills:
        roadmap_items = []
        resource_map = {
            "python": "Python for Everybody (Coursera) or Real Python (realpython.com)",
            "machine learning": "Andrew Ng's Machine Learning Specialization (Coursera)",
            "deep learning": "Deep Learning Specialization — deeplearning.ai",
            "sql": "Mode Analytics SQL Tutorial (mode.com/sql-tutorial)",
            "docker": "Docker Mastery (Udemy — Bret Fisher)",
            "aws": "AWS Cloud Practitioner Essentials (AWS Training)",
            "kubernetes": "Kubernetes for Beginners (KodeKloud)",
            "react": "The Complete React Developer Course (Udemy — Maximilian Schwarzmüller)",
            "tensorflow": "TensorFlow Developer Certificate (Google)",
            "pytorch": "PyTorch for Deep Learning (fast.ai)",
        }
        for skill in missing_skills[:5]:
            resource = resource_map.get(
                skill.lower(),
                f"Search \"{skill} tutorial\" on Coursera, Udemy, or official documentation.",
            )
            roadmap_items.append(f"- **{skill.title()}**: {resource}")
        roadmap_section += "\n".join(roadmap_items)
    else:
        roadmap_section += "Your current skill set aligns well. Focus on deepening expertise in matched areas."

    projects_section = (
        "### 4. Suggested Projects\n"
        "Build these projects to strengthen your portfolio for this role:\n"
    )
    if missing_skills:
        skill_str = ", ".join(missing_skills[:3])
        projects_section += (
            f"1. **End-to-End Project** — Build a project incorporating: {skill_str}.\n"
            "2. **Open Source Contribution** — Contribute to a GitHub repo related to your gaps.\n"
            "3. **Portfolio Showcase** — Create a documented case study demonstrating your existing strengths."
        )
    else:
        projects_section += (
            "1. **Advanced Portfolio Project** — Combine your top skills into a complex, deployed application.\n"
            "2. **Technical Blog** — Write about your best projects to demonstrate thought leadership.\n"
            "3. **Open Source Contribution** — Contribute to projects used by the target company."
        )

    strategy_section = (
        "### 5. Application Strategy\n"
        f"{readiness}\n\n"
        "**Action items:**\n"
        "- Tailor your resume header and summary to mirror the JD's language.\n"
        "- Quantify achievements wherever possible (e.g., 'improved model accuracy by 15%').\n"
        "- Address skill gaps proactively in your cover letter by highlighting fast learning ability.\n"
        "- Network with employees at the target company via LinkedIn to gain referrals.\n"
        "- Prepare to discuss missing skills honestly while demonstrating a learning roadmap in interviews."
    )

    report = "\n\n".join([
        strengths_section,
        gaps_section,
        roadmap_section,
        projects_section,
        strategy_section,
    ])

    return report


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def generate_career_advice(
    resume_skills: list[str],
    jd_skills: list[str],
    matching_skills: list[str],
    missing_skills: list[str],
    extra_skills: list[str],
    match_score: float,
    doc_similarity: float,
    groq_api_key: Optional[str] = None,
    groq_model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
) -> dict:
    """
    Generate career advice using Groq (Llama 4) or rule-based fallback.

    Priority:
        1. Groq API (GROQ_API_KEY env var or groq_api_key param)
        2. Rule-based fallback

    Args:
        resume_skills: Extracted resume skills.
        jd_skills: Extracted JD skills.
        matching_skills: Skills in both.
        missing_skills: JD skills absent from resume.
        extra_skills: Resume-only skills.
        match_score: Final weighted match percentage.
        doc_similarity: Document similarity percentage.
        groq_api_key: Optional Groq API key. Falls back to env var GROQ_API_KEY.
        groq_model: Groq model to use.

    Returns:
        Dictionary with:
            - 'advice': str — formatted markdown career advice
            - 'source': str — 'groq' or 'rule-based'
    """
    # Build prompt (shared for both LLM providers)
    prompt = _build_prompt(
        resume_skills=resume_skills,
        jd_skills=jd_skills,
        matching_skills=matching_skills,
        missing_skills=missing_skills,
        extra_skills=extra_skills,
        match_score=match_score,
        doc_similarity=doc_similarity,
    )

    # ---- 1. Try Groq ----
    resolved_groq_key = groq_api_key or os.getenv("GROQ_API_KEY", "").strip()
    if resolved_groq_key:
        try:
            logger.info("Calling Groq API for career advice (model=%s)...", groq_model)
            advice = _call_groq_api(prompt, api_key=resolved_groq_key, model=groq_model)
            return {"advice": advice, "source": "groq"}
        except Exception as exc:
            logger.warning("Groq call failed, falling back to rule-based: %s", exc)

    # ---- 2. Rule-based fallback ----
    logger.info("Generating rule-based career advice.")
    advice = _rule_based_advice(
        resume_skills=resume_skills,
        jd_skills=jd_skills,
        matching_skills=matching_skills,
        missing_skills=missing_skills,
        extra_skills=extra_skills,
        match_score=match_score,
    )
    return {"advice": advice, "source": "rule-based"}
