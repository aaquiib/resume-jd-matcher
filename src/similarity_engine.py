"""
similarity_engine.py
--------------------
Computes semantic similarity between texts using Sentence-BERT embeddings
and cosine similarity from scikit-learn.
"""

import logging
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# Default model — lightweight yet powerful
DEFAULT_MODEL = "all-MiniLM-L6-v2"


class SimilarityEngine:
    """
    Computes semantic similarity between documents and skill lists
    using Sentence-BERT embeddings.

    Attributes:
        model: Loaded SentenceTransformer model.
        model_name: Name of the model in use.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        """
        Load the Sentence-BERT model.

        Args:
            model_name: HuggingFace model identifier for SentenceTransformer.
        """
        logger.info("Loading SentenceTransformer model: %s", model_name)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        logger.info("SentenceTransformer model loaded.")

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Generate sentence embeddings for a list of texts.

        Args:
            texts: List of string inputs to embed.

        Returns:
            NumPy array of shape (N, embedding_dim).
        """
        if not texts:
            return np.array([])
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def document_similarity(self, text_a: str, text_b: str) -> float:
        """
        Compute cosine similarity between two documents.

        Chunks long documents into sentences and uses mean pooling for
        a more robust similarity estimate.

        Args:
            text_a: First document string.
            text_b: Second document string.

        Returns:
            Cosine similarity score in [0, 1].
        """
        embeddings = self.embed([text_a, text_b])
        sim = cosine_similarity(embeddings[0:1], embeddings[1:2])
        return float(np.clip(sim[0][0], 0.0, 1.0))

    def skill_list_similarity(
        self,
        skills_a: list[str],
        skills_b: list[str],
    ) -> float:
        """
        Compute average pairwise cosine similarity between two skill sets.

        Each skill in list A is matched against the closest skill in list B
        and the mean of max similarities is returned (asymmetric recall).

        Args:
            skills_a: Candidate resume skills.
            skills_b: Job description skills.

        Returns:
            Mean-of-max similarity in [0, 1].  Returns 0.0 if either list is empty.
        """
        if not skills_a or not skills_b:
            return 0.0

        emb_a = self.embed(skills_a)
        emb_b = self.embed(skills_b)

        # Shape: (len_a, len_b)
        sim_matrix = cosine_similarity(emb_a, emb_b)

        # For each skill in B (JD), find the best match from A (resume)
        max_per_jd_skill = sim_matrix.max(axis=0)
        return float(np.mean(max_per_jd_skill))

    def compute_all_similarities(
        self,
        resume_text: str,
        jd_text: str,
        resume_skills: list[str],
        jd_skills: list[str],
    ) -> dict:
        """
        Compute all similarity metrics in one call.

        Args:
            resume_text: Full resume text.
            jd_text: Full job description text.
            resume_skills: Extracted resume skills.
            jd_skills: Extracted JD skills.

        Returns:
            Dictionary with:
                - 'doc_similarity': float, overall document similarity
                - 'skill_similarity': float, semantic skill overlap
        """
        doc_sim = self.document_similarity(resume_text, jd_text)
        skill_sim = self.skill_list_similarity(resume_skills, jd_skills)

        logger.info(
            "Similarities — doc: %.4f | skill: %.4f",
            doc_sim,
            skill_sim,
        )

        return {
            "doc_similarity": doc_sim,
            "skill_similarity": skill_sim,
        }
