# AI Resume-Job Description Matcher & Career Advisor

A production-grade NLP pipeline that quantifies resume-to-job-description alignment using semantic embeddings, phrase-level skill extraction, and LLM-generated career guidance — all served through an interactive Streamlit dashboard.

---

## Overview

Traditional keyword-based resume screeners fail to capture semantic equivalence between different phrasings of the same skill or context. This system addresses that limitation by combining **transformer-based sentence embeddings** with a **curated skill taxonomy** and a **weighted scoring model** to produce an interpretable match score alongside a personalized career roadmap.

The pipeline is designed around a clear separation of concerns — PDF parsing, skill extraction, similarity computation, and scoring each live in isolated modules — making it easy to swap components or extend the taxonomy without touching the rest of the system.

---

## Features

- **Semantic Document Similarity** — Full-document SBERT embeddings with cosine similarity, capturing meaning beyond surface-level token overlap
- **Phrase-Level Skill Extraction** — spaCy `PhraseMatcher` against a hand-curated taxonomy of 250+ skills across 11 technical and professional categories
- **Weighted Match Scoring** — Composite score combining document semantics, skill overlap ratio, and exact skill coverage
- **Skill Gap Analysis** — Explicit diff of matching, missing, and extra skills with category-level breakdowns
- **AI Career Advisor** — Structured learning roadmap and application strategy via Groq's Llama 4 API, with rule-based fallback when no API key is configured
- **Export** — Download match results as CSV or career advice as plain text

---

## Architecture

```
User uploads Resume PDF + Job Description PDF
              │
              ▼
       ┌─────────────┐
       │  PDF Parser  │  PyMuPDF extraction + regex normalization
       └──────┬──────┘
              │
              ▼
       ┌─────────────────────────────────────┐
       │            NLP Pipeline             │
       │                                     │
       │  ┌──────────────────────────────┐   │
       │  │      Skill Extractor         │   │
       │  │  spaCy PhraseMatcher         │   │
       │  │  SKILL_TAXONOMY (250+ terms) │   │
       │  └──────────────────────────────┘   │
       │                                     │
       │  ┌──────────────────────────────┐   │
       │  │      Similarity Engine       │   │
       │  │  Sentence-BERT embeddings    │   │
       │  │  Cosine similarity (sklearn) │   │
       │  │  Mean-of-max skill matching  │   │
       │  └──────────────────────────────┘   │
       │                                     │
       │  ┌──────────────────────────────┐   │
       │  │         Scoring              │   │
       │  │  Weighted composite formula  │   │
       │  └──────────────────────────────┘   │
       └─────────────────────────────────────┘
              │
              ▼
       ┌─────────────┐
       │ LLM Advisor │  Groq API (Llama 4) → rule-based fallback
       └──────┬──────┘
              │
              ▼
       ┌─────────────────┐
       │  Streamlit UI   │  Dashboard, skill tabs, export
       └─────────────────┘
```

---

## NLP Design Decisions

### Skill Extraction — Why PhraseMatcher over NER?

Off-the-shelf Named Entity Recognition models are trained on general corpora (news, Wikipedia) and do not reliably detect technical skill phrases like `"FastAPI"`, `"dbt"`, or `"vector databases"`. A domain-specific `PhraseMatcher` over a curated taxonomy gives deterministic, high-precision extraction with no inference overhead.

The taxonomy covers 11 categories:

| Category | Examples |
|---|---|
| Programming Languages | Python, Rust, Go, TypeScript |
| ML / AI | PyTorch, Transformers, RAG, Hugging Face |
| Web Frameworks | FastAPI, Django, Next.js, Spring Boot |
| Data Tools | Spark, Airflow, dbt, Tableau |
| Databases | PostgreSQL, Redis, Elasticsearch, Pinecone |
| Cloud / DevOps | AWS, Kubernetes, Terraform, CI/CD |
| Data Science Methods | Regression, Clustering, A/B Testing |
| API & Auth | REST, GraphQL, OAuth, JWT |
| Architecture Patterns | Microservices, Event-driven, Message Queues |
| Version Control | Git, GitHub, GitLab |
| Soft Skills | Communication, Leadership, Agile, Scrum |

### Semantic Similarity — Sentence-BERT

The system uses `all-MiniLM-L6-v2` from the `sentence-transformers` library to encode both documents into 384-dimensional dense vectors. Cosine similarity on these embeddings captures paraphrastic equivalences that bag-of-words approaches miss — e.g., "built REST APIs" and "developed web services" will embed close together.

Skill-list similarity uses an **asymmetric mean-of-max** strategy: for each JD skill embedding, find the maximum cosine score against any resume skill embedding, then average across all JD skills. This rewards partial semantic coverage without penalizing extra skills on the resume.

### Scoring Formula

```
Match Score = 0.60 × Document Similarity
            + 0.30 × Skill Overlap Ratio
            + 0.10 × Exact Skill Bonus
```

- **Document Similarity (60%)** — Captures overall contextual alignment; a candidate describing experience in adjacent roles can still score well even with partial skill overlap
- **Skill Overlap Ratio (30%)** — `|resume_skills ∩ jd_skills| / |jd_skills|`, directly measures JD coverage
- **Exact Skill Bonus (10%)** — Rewards verbatim taxonomy matches, which are more signal-dense than semantic proximity alone

Weights were chosen to balance semantic richness against concrete skill coverage, with document context carrying the majority of the signal.

---

## Tech Stack

| Component | Technology |
|---|---|
| NLP / Phrase Matching | spaCy 3.7+ (`en_core_web_sm`) |
| Semantic Embeddings | Sentence-BERT (`all-MiniLM-L6-v2`) |
| Similarity Computation | scikit-learn cosine similarity |
| PDF Extraction | PyMuPDF (fitz) |
| LLM Career Advice | Groq API — Llama 4 Scout / Maverick |
| Data Handling | Pandas, NumPy |
| Frontend | Streamlit |
| Environment Config | python-dotenv |

---

## Project Structure

```
resume-job-matching/
├── app.py                  # Streamlit entry point
├── requirements.txt
├── setup.bat               # Windows one-click setup
├── .env.example
└── src/
    ├── pdf_parser.py       # PyMuPDF extraction + text normalization
    ├── nlp_pipeline.py     # Pipeline orchestration, model caching
    ├── skill_extractor.py  # PhraseMatcher + SKILL_TAXONOMY
    ├── similarity_engine.py# SBERT embeddings + cosine similarity
    ├── scoring.py          # Weighted scoring + MatchResult dataclass
    └── llm_advisor.py      # Groq API integration + rule-based fallback
```

---

## Setup

**Prerequisites:** Python 3.8+

### Windows (automated)

```bash
setup.bat
```

This will create a virtual environment, install all dependencies, and download the required spaCy model.

### Manual

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux / macOS

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Environment Configuration

```bash
cp .env.example .env
```

Open `.env` and add your Groq API key:

```
GROQ_API_KEY="gsk_your_key_here"
```

> The Groq API key is optional. If not provided, the system falls back to a rule-based career advisor that generates structured advice without an LLM.

---

## Usage

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

1. Upload your resume as a PDF
2. Upload the job description as a PDF
3. Click **Analyze Match**
4. Review the match score, skill gap breakdown, and AI career advice
5. Export results as CSV or the advice as a text file

---

## Output Breakdown

| Metric | Description |
|---|---|
| Match Score | Weighted composite score (0–100%) |
| Document Similarity | SBERT cosine similarity of full documents |
| Skill Similarity | Mean-of-max semantic skill overlap |
| Skill Overlap Ratio | Exact + fuzzy matched JD skills / total JD skills |
| Matching Skills | Skills present in both resume and JD |
| Missing Skills | JD skills not found in resume — primary skill gaps |
| Extra Skills | Resume skills not required by JD |

---

## Model Details

| Model | Size | Purpose |
|---|---|---|
| `en_core_web_sm` (spaCy) | ~40 MB | Tokenization, phrase boundary detection |
| `all-MiniLM-L6-v2` (SBERT) | ~80 MB | 384-dim sentence embeddings for semantic similarity |
| Llama 4 Scout 17B (Groq) | Hosted API | Career advice, skill gap analysis, learning roadmap |

Both local models are downloaded on first run and cached automatically.

---

## Limitations

- Skill taxonomy coverage is finite — niche or emerging tools may not be recognized unless added to `SKILL_TAXONOMY` in `src/skill_extractor.py`
- PDF extraction quality depends on the document's structure; scanned/image PDFs without embedded text will not parse correctly
- Document similarity scores can be high for broadly related resumes even when specific required skills are absent — always review the skill gap tab alongside the overall score
- The LLM advisor prompt is English-only; multilingual resumes are not currently supported

---

## Extending the Taxonomy

To add new skills, edit `SKILL_TAXONOMY` in `src/skill_extractor.py`:

```python
SKILL_TAXONOMY = {
    "your_category": [
        "New Skill",
        "Another Tool",
        ...
    ],
    ...
}
```

No retraining or model changes required — the PhraseMatcher is rebuilt at startup.

---

## Requirements

```
streamlit>=1.32.0
PyMuPDF>=1.23.0
spacy>=3.7.0
sentence-transformers>=2.7.0
scikit-learn>=1.4.0
numpy>=1.26.0
pandas>=2.2.0
groq>=0.9.0
python-dotenv>=1.0.0
```

---

## License

MIT
