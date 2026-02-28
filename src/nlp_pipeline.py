"""
nlp_pipeline.py
---------------
Orchestrates the full NLP processing pipeline:
  1. Load spaCy model (singleton)
  2. Load SentenceTransformer (singleton)
  3. Run skill extraction on resume + JD text
  4. Compute all similarity metrics

Designed as a stateful pipeline that caches heavy models after first load
to avoid reloading on every Streamlit rerun.
"""

import logging
from functools import lru_cache
from typing import Optional

import spacy

from src.skill_extractor import SkillExtractor
from src.similarity_engine import SimilarityEngine

logger = logging.getLogger(__name__)

SPACY_MODEL = "en_core_web_sm"
SBERT_MODEL = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _load_spacy(model_name: str = SPACY_MODEL) -> spacy.language.Language:
    """Load and cache the spaCy model."""
    try:
        logger.info("Loading spaCy model: %s", model_name)
        nlp = spacy.load(model_name)
        logger.info("spaCy model loaded.")
        return nlp
    except OSError:
        raise RuntimeError(
            f"spaCy model '{model_name}' not found. "
            f"Run: python -m spacy download {model_name}"
        )


@lru_cache(maxsize=1)
def _load_sbert(model_name: str = SBERT_MODEL) -> SimilarityEngine:
    """Load and cache the SimilarityEngine."""
    return SimilarityEngine(model_name=model_name)


class NLPPipeline:
    """
    Full NLP analysis pipeline for resume vs. job description comparison.

    This class coordinates:
        - Skill extraction via SkillExtractor
        - Semantic similarity via SimilarityEngine

    Attributes:
        skill_extractor: SkillExtractor instance.
        similarity_engine: SimilarityEngine instance.
    """

    def __init__(
        self,
        spacy_model: str = SPACY_MODEL,
        sbert_model: str = SBERT_MODEL,
    ) -> None:
        """
        Initialize the pipeline, loading models if not already cached.

        Args:
            spacy_model: spaCy model name.
            sbert_model: SentenceTransformer model name.
        """
        nlp = _load_spacy(spacy_model)
        self.skill_extractor = SkillExtractor(nlp)
        self.similarity_engine = _load_sbert(sbert_model)

    def analyze(self, resume_text: str, jd_text: str) -> dict:
        """
        Run the full analysis pipeline.

        Args:
            resume_text: Normalized resume text.
            jd_text: Normalized job description text.

        Returns:
            Dictionary with:
                - 'resume_skills': list of extracted resume skills
                - 'jd_skills': list of extracted JD skills
                - 'resume_skill_categories': dict of category -> skills (resume)
                - 'jd_skill_categories': dict of category -> skills (JD)
                - 'doc_similarity': float [0,1]
                - 'skill_similarity': float [0,1]
        """
        logger.info("Running skill extraction on resume...")
        resume_extraction = self.skill_extractor.extract(resume_text)

        logger.info("Running skill extraction on JD...")
        jd_extraction = self.skill_extractor.extract(jd_text)

        resume_skills: list[str] = resume_extraction["skills"]
        jd_skills: list[str] = jd_extraction["skills"]

        logger.info(
            "Extracted %d resume skills and %d JD skills.",
            len(resume_skills),
            len(jd_skills),
        )

        logger.info("Computing semantic similarities...")
        similarities = self.similarity_engine.compute_all_similarities(
            resume_text=resume_text,
            jd_text=jd_text,
            resume_skills=resume_skills,
            jd_skills=jd_skills,
        )

        return {
            "resume_skills": resume_skills,
            "jd_skills": jd_skills,
            "resume_skill_categories": resume_extraction["skill_categories"],
            "jd_skill_categories": jd_extraction["skill_categories"],
            "doc_similarity": similarities["doc_similarity"],
            "skill_similarity": similarities["skill_similarity"],
        }
