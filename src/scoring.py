"""
scoring.py
----------
Implements the weighted match scoring algorithm.

Final Match Score =
    0.60 * doc_similarity
  + 0.30 * skill_overlap_ratio
  + 0.10 * exact_skill_bonus

Where:
  - doc_similarity  : cosine similarity of SBERT document embeddings [0,1]
  - skill_overlap   : |resume_skills ∩ jd_skills| / |jd_skills| [0,1]
  - exact_skill_bonus: fraction of JD skills found verbatim in resume [0,1]

The score is clipped to [0, 100] and expressed as a percentage.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Weighting constants — must sum to 1.0
WEIGHT_DOC_SIMILARITY: float = 0.60
WEIGHT_SKILL_OVERLAP: float = 0.30
WEIGHT_EXACT_BONUS: float = 0.10


@dataclass
class MatchResult:
    """
    Structured container for all scoring outputs.

    Attributes:
        match_score: Final weighted match percentage [0, 100].
        doc_similarity: Raw document cosine similarity [0, 1].
        skill_similarity: Semantic skill similarity [0, 1].
        skill_overlap_ratio: Exact overlap ratio [0, 1].
        exact_skill_bonus: Bonus for verbatim matches [0, 1].
        matching_skills: Skills present in both resume and JD.
        missing_skills: JD skills absent from resume.
        extra_skills: Resume skills not required by JD.
        component_scores: Dict of individual weighted contributions.
    """
    match_score: float
    doc_similarity: float
    skill_similarity: float
    skill_overlap_ratio: float
    exact_skill_bonus: float
    matching_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    extra_skills: list[str] = field(default_factory=list)
    component_scores: dict = field(default_factory=dict)


def compute_skill_overlap(
    resume_skills: list[str],
    jd_skills: list[str],
) -> tuple[float, float, list[str], list[str], list[str]]:
    """
    Compute exact skill overlap statistics between resume and JD skill sets.

    Args:
        resume_skills: Skills extracted from resume.
        jd_skills: Skills extracted from JD.

    Returns:
        Tuple of:
            - skill_overlap_ratio: |intersection| / |jd_skills|
            - exact_skill_bonus: same as overlap_ratio (reserved for bonus calc)
            - matching_skills: sorted list of shared skills
            - missing_skills: JD skills not in resume
            - extra_skills: Resume skills not in JD
    """
    resume_set = {s.lower().strip() for s in resume_skills}
    jd_set = {s.lower().strip() for s in jd_skills}

    matching = sorted(resume_set & jd_set)
    missing = sorted(jd_set - resume_set)
    extra = sorted(resume_set - jd_set)

    if not jd_set:
        overlap_ratio = 0.0
        bonus = 0.0
    else:
        overlap_ratio = len(matching) / len(jd_set)
        # Bonus: weighted by how many JD skills are exact matches (same value here,
        # could be extended to weight critical skills higher)
        bonus = overlap_ratio

    return overlap_ratio, bonus, matching, missing, extra


def compute_match_score(
    doc_similarity: float,
    skill_similarity: float,
    resume_skills: list[str],
    jd_skills: list[str],
) -> MatchResult:
    """
    Compute the final weighted match score.

    Formula:
        score = 0.60 * doc_similarity
              + 0.30 * skill_overlap_ratio
              + 0.10 * exact_skill_bonus

    Args:
        doc_similarity: Document-level cosine similarity [0, 1].
        skill_similarity: Semantic skill similarity [0, 1] (used for display).
        resume_skills: Extracted resume skills.
        jd_skills: Extracted JD skills.

    Returns:
        MatchResult dataclass with all metrics populated.
    """
    overlap_ratio, exact_bonus, matching, missing, extra = compute_skill_overlap(
        resume_skills, jd_skills
    )

    weighted_doc = WEIGHT_DOC_SIMILARITY * doc_similarity
    weighted_overlap = WEIGHT_SKILL_OVERLAP * overlap_ratio
    weighted_bonus = WEIGHT_EXACT_BONUS * exact_bonus

    raw_score = weighted_doc + weighted_overlap + weighted_bonus
    # Clip to [0, 1] then convert to percentage
    final_score = round(min(max(raw_score, 0.0), 1.0) * 100, 1)

    component_scores = {
        "doc_similarity_contribution": round(weighted_doc * 100, 2),
        "skill_overlap_contribution": round(weighted_overlap * 100, 2),
        "exact_bonus_contribution": round(weighted_bonus * 100, 2),
    }

    logger.info(
        "Match Score: %.1f%% | doc=%.3f | overlap=%.3f | bonus=%.3f",
        final_score,
        doc_similarity,
        overlap_ratio,
        exact_bonus,
    )

    return MatchResult(
        match_score=final_score,
        doc_similarity=round(doc_similarity * 100, 2),
        skill_similarity=round(skill_similarity * 100, 2),
        skill_overlap_ratio=round(overlap_ratio * 100, 2),
        exact_skill_bonus=round(exact_bonus * 100, 2),
        matching_skills=matching,
        missing_skills=missing,
        extra_skills=extra,
        component_scores=component_scores,
    )


def score_label(score: float) -> tuple[str, str]:
    """
    Return a human-readable label and color hex for a match score.

    Args:
        score: Match percentage [0, 100].

    Returns:
        Tuple of (label, color_hex).
    """
    if score >= 80:
        return "Excellent Match", "#22c55e"   # green
    elif score >= 65:
        return "Good Match", "#84cc16"        # lime
    elif score >= 50:
        return "Moderate Match", "#f59e0b"    # amber
    elif score >= 35:
        return "Weak Match", "#ef4444"        # red
    else:
        return "Poor Match", "#dc2626"        # dark red
