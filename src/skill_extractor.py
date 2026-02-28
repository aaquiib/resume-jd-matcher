"""
skill_extractor.py
------------------
Extracts technical and professional skills from text using:
  - A curated skill taxonomy (case-insensitive phrase matching)
  - spaCy NER for additional entity detection

The taxonomy covers: programming languages, frameworks, databases, cloud,
ML/AI, DevOps, soft skills, data tools, and common role-related terms.
"""

import re
import logging
from typing import Optional

import spacy
from spacy.matcher import PhraseMatcher

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Skill Taxonomy
# ---------------------------------------------------------------------------

SKILL_TAXONOMY: dict[str, list[str]] = {
    "programming_languages": [
        "python", "java", "javascript", "typescript", "c++", "c#", "c",
        "r", "go", "golang", "rust", "kotlin", "swift", "scala", "ruby",
        "php", "perl", "bash", "shell", "powershell", "matlab", "julia",
        "lua", "dart", "groovy", "haskell", "elixir", "clojure",
    ],
    "web_frameworks": [
        "react", "reactjs", "react.js", "angular", "angularjs", "vue",
        "vue.js", "vuejs", "next.js", "nextjs", "nuxt.js", "svelte",
        "django", "flask", "fastapi", "spring", "spring boot", "express",
        "express.js", "node.js", "nodejs", "ruby on rails", "rails",
        "laravel", "asp.net", "blazor", "htmx", "streamlit", "gradio",
    ],
    "ml_ai": [
        "machine learning", "deep learning", "neural networks",
        "natural language processing", "nlp", "computer vision", "cv",
        "reinforcement learning", "transfer learning", "fine-tuning",
        "llm", "large language models", "generative ai", "genai",
        "pytorch", "tensorflow", "keras", "scikit-learn", "sklearn",
        "xgboost", "lightgbm", "catboost", "hugging face", "transformers",
        "openai", "langchain", "llamaindex", "llama", "gpt", "bert",
        "sentence-transformers", "spacy", "nltk", "gensim",
        "stable diffusion", "diffusion models", "rag", "vector database",
        "feature engineering", "hyperparameter tuning", "model deployment",
        "mlops", "mlflow", "wandb", "dvc",
    ],
    "data_tools": [
        "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly",
        "bokeh", "tableau", "power bi", "looker", "airflow", "luigi",
        "spark", "pyspark", "hadoop", "hive", "kafka", "dbt",
        "excel", "google sheets", "jupyter", "colab",
    ],
    "databases": [
        "sql", "mysql", "postgresql", "postgres", "sqlite", "oracle",
        "sql server", "mssql", "nosql", "mongodb", "redis", "cassandra",
        "dynamodb", "elasticsearch", "neo4j", "pinecone", "weaviate",
        "chroma", "faiss", "qdrant", "supabase", "firebase",
    ],
    "cloud_devops": [
        "aws", "amazon web services", "azure", "microsoft azure",
        "gcp", "google cloud", "docker", "kubernetes", "k8s",
        "terraform", "ansible", "jenkins", "ci/cd", "github actions",
        "gitlab ci", "circleci", "helm", "serverless", "lambda",
        "ec2", "s3", "rds", "sagemaker", "vertex ai", "azure ml",
        "linux", "unix", "nginx", "apache",
    ],
    "version_control": [
        "git", "github", "gitlab", "bitbucket", "svn",
    ],
    "soft_skills": [
        "communication", "leadership", "teamwork", "problem solving",
        "critical thinking", "time management", "collaboration",
        "project management", "agile", "scrum", "kanban", "jira",
        "presentation", "stakeholder management", "mentoring",
    ],
    "data_science_methods": [
        "regression", "classification", "clustering", "dimensionality reduction",
        "pca", "a/b testing", "statistical analysis", "bayesian",
        "time series", "forecasting", "anomaly detection", "recommendation systems",
    ],
    "api_integration": [
        "rest", "restful", "graphql", "grpc", "soap", "api", "webhooks",
        "oauth", "jwt", "openapi", "swagger",
    ],
    "architecture": [
        "microservices", "monolith", "event-driven", "message queue",
        "rabbitmq", "celery", "distributed systems", "high availability",
        "load balancing", "caching", "cdn",
    ],
}

# Flatten to a set for lookups, preserving original casing as well
_ALL_SKILLS_LOWER: dict[str, str] = {}  # lowercase -> canonical form

for _category, _skills in SKILL_TAXONOMY.items():
    for _skill in _skills:
        _ALL_SKILLS_LOWER[_skill.lower()] = _skill


def _build_phrase_matcher(nlp: spacy.language.Language) -> PhraseMatcher:
    """Build a spaCy PhraseMatcher from the skill taxonomy."""
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    for category, skills in SKILL_TAXONOMY.items():
        patterns = [nlp.make_doc(skill) for skill in skills]
        matcher.add(category, patterns)
    return matcher


class SkillExtractor:
    """
    Extracts skills from text using a PhraseMatcher over a curated taxonomy.

    Attributes:
        nlp: Loaded spaCy language model.
        matcher: Compiled PhraseMatcher.
    """

    def __init__(self, nlp: spacy.language.Language) -> None:
        """
        Initialize extractor with a spaCy model.

        Args:
            nlp: A loaded spaCy Language object (e.g., en_core_web_sm).
        """
        self.nlp = nlp
        self.matcher = _build_phrase_matcher(nlp)
        logger.info("SkillExtractor initialized with %d skill entries.", len(_ALL_SKILLS_LOWER))

    def extract(self, text: str) -> dict:
        """
        Extract skills from text.

        Args:
            text: Normalized input text.

        Returns:
            Dictionary with:
                - 'skills': sorted list of unique detected skills (canonical form)
                - 'skill_categories': dict mapping category -> list of skills
        """
        doc = self.nlp(text.lower())
        matches = self.matcher(doc)

        found_skills: set[str] = set()
        category_map: dict[str, list[str]] = {}

        for match_id, start, end in matches:
            category = self.nlp.vocab.strings[match_id]
            span_text = doc[start:end].text  # already lowercase due to LOWER attr
            canonical = _ALL_SKILLS_LOWER.get(span_text, span_text)
            found_skills.add(canonical)
            category_map.setdefault(category, [])
            if canonical not in category_map[category]:
                category_map[category].append(canonical)

        return {
            "skills": sorted(found_skills),
            "skill_categories": category_map,
        }

    def normalize_skill(self, skill: str) -> str:
        """
        Normalize a skill string (lowercase lookup in taxonomy).

        Args:
            skill: Raw skill string.

        Returns:
            Canonical skill name or original if not found.
        """
        return _ALL_SKILLS_LOWER.get(skill.strip().lower(), skill.strip().lower())
