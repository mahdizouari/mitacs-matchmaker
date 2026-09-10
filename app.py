import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Mitacs AI Matchmaker", layout="wide")
st.title("🎯 Mitacs Project Matchmaker")
st.markdown(
    "Reads your full CV, compares it against **Required Skills, Student Roles, "
    "Project Activities, Project Description, and Additional Information**, "
    "and reranks the top candidates for precision."
)

PROJECTS_CSV = "mitacs_projects.csv"

FIELD_COLUMNS = [
    "Required_Skills",
    "Student_Roles",
    "Project_Activities",
    "Project_Description",
    "Additional_Information",
]

# How much each field counts toward the final score. Skills/Roles matter
# most for whether you're actually a fit; Additional_Information tends to
# be logistics/context, so it counts least. Must sum to 1.0.
FIELD_WEIGHTS = {
    "Required_Skills": 0.35,
    "Student_Roles": 0.20,
    "Project_Activities": 0.15,
    "Project_Description": 0.20,
    "Additional_Information": 0.10,
}

# Blend between semantic (embedding) similarity and exact keyword overlap.
# 0.7 semantic / 0.3 keyword tends to reward real conceptual fit while still
# giving credit for literal tool/tech matches embeddings can under-weight.
SEMANTIC_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3

# How many bi-encoder candidates get promoted to the slower, more precise
# cross-encoder reranking stage.
RERANK_POOL_SIZE = 20

# Final blend between the reranker and the bi-encoder score, so one odd
# reranker score can't completely override everything else.
RERANK_WEIGHT = 0.65
BIENCODER_WEIGHT = 0.35

# A reasonably broad tech/skills gazetteer used for deterministic keyword
# matching, on top of whatever the user types in manually below.
DEFAULT_TECH_KEYWORDS = [
    "python", "java", "c++", "c#", "javascript", "typescript", "sql", "r",
    "machine learning", "deep learning", "artificial intelligence", "nlp",
    "computer vision", "reinforcement learning", "data mining",
    "anomaly detection", "time series", "forecasting",
    "pytorch", "tensorflow", "scikit-learn", "keras", "lightgbm", "xgboost",
    "pandas", "numpy",
    "docker", "kubernetes", "aws", "azure", "gcp", "ci/cd", "git", "linux",
    "postgresql", "mongodb", "mysql", "etl", "data pipeline", "data warehouse",
    "power bi", "spark", "hadoop",
    "flutter", "android", "ios", "react", "angular", "spring boot",
    "fastapi", "rest api", "microservices",
    "embedded systems", "real-time systems", "signal processing",
]


# ============================================================
# MODELS (cached so they load once per session, not per click)
# ============================================================
@st.cache_resource
def load_bi_encoder():
    # Multilingual: handles projects written in French as well as English,
    # instead of penalizing them just for language mismatch.
    return SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")


@st.cache_resource
def load_cross_encoder():
    # Multilingual reranker for the final precision pass on the top
    # candidates only (keeps this fast despite being a heavier model).
    return CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")


# ============================================================
# DATA LOADING (cache-busted on the CSV's file modification time,
# so re-running the scraper actually refreshes results here)
# ============================================================
@st.cache_data
def load_projects(_mtime):
    if not os.path.exists(PROJECTS_CSV):
        return None, f"Error: '{PROJECTS_CSV}' not found. Please run the scraper first."

    df = pd.read_csv(PROJECTS_CSV)
    for col in FIELD_COLUMNS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")

    df["Combined_Text"] = df[FIELD_COLUMNS].agg(" | ".join, axis=1)
    return df, "Success"


@st.cache_data
def embed_project_fields(_mtime, field_texts_by_column):
    """Embed each field column separately (not concatenated) so we can score
    and weight them individually instead of losing that signal in one blob."""
    model = load_bi_encoder()
    embeddings = {}
    for col, texts in field_texts_by_column.items():
        embeddings[col] = model.encode(texts, show_progress_bar=False)
    return embeddings


# ============================================================
# CV PARSING
# ============================================================
def extract_cv_text(cv_file) -> str:
    text = ""
    try:
        with pdfplumber.open(cv_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception:
        pass

    if not text.strip():
        # Fallback path if pdfplumber fails on an unusual PDF encoding.
        try:
            import PyPDF2
            cv_file.seek(0)
            reader = PyPDF2.PdfReader(cv_file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception:
            pass

    return text


# Common resume section headers, used to split the CV into meaningful chunks
# rather than an arbitrary sliding window, when possible.
SECTION_HEADERS = [
    "education", "work experience", "experience", "projects", "project",
    "technical skills", "skills", "languages", "certifications",
    "publications", "research", "summary", "objective",
]


def chunk_cv_text(cv_text: str) -> list[str]:
    """Split the CV into section-based chunks so the embedding model actually
    sees the whole document instead of silently truncating it at ~256-384
    tokens. Falls back to a fixed-size sliding window if no headers match."""
    lines = cv_text.split("\n")
    chunks = []
    current_chunk_lines = []

    def is_header(line: str) -> bool:
        stripped = line.strip().lower().strip(":")
        return any(stripped == h or stripped.startswith(h) for h in SECTION_HEADERS) and len(stripped) < 40

    for line in lines:
        if is_header(line) and current_chunk_lines:
            chunks.append("\n".join(current_chunk_lines).strip())
            current_chunk_lines = [line]
        else:
            current_chunk_lines.append(line)
    if current_chunk_lines:
        chunks.append("\n".join(current_chunk_lines).strip())

    chunks = [c for c in chunks if len(c.strip()) > 20]

    if len(chunks) <= 1:
        # No clear headers found - fall back to an overlapping word-window
        # split so nothing past the token limit gets silently dropped.
        words = cv_text.split()
        window, overlap = 180, 40
        chunks = []
        i = 0
        while i < len(words):
            chunks.append(" ".join(words[i:i + window]))
            i += window - overlap
        chunks = [c for c in chunks if c.strip()]

    return chunks if chunks else [cv_text]


def extract_matched_keywords(cv_text: str, project_text: str, extra_keywords: list[str]) -> tuple[float, list[str]]:
    """Deterministic keyword-overlap score: catches exact tool/tech matches
    that a semantic embedding can sometimes under-weight."""
    vocab = set(k.strip().lower() for k in DEFAULT_TECH_KEYWORDS + extra_keywords if k.strip())
    cv_lower = cv_text.lower()
    project_lower = project_text.lower()

    matched = [kw for kw in vocab if kw in cv_lower and kw in project_lower]
    project_kw_count = sum(1 for kw in vocab if kw in project_lower)

    if project_kw_count == 0:
        return 0.0, matched
    score = len(matched) / project_kw_count
    return min(score, 1.0), matched


# ============================================================
# UI
# ============================================================
st.markdown("### 1. Upload Your CV")
cv_file = st.file_uploader("Upload your CV (PDF)", type=["pdf"])

st.markdown("### 2. Extra keywords to emphasize (optional)")
user_keywords_input = st.text_input(
    "Comma-separated. These are added on top of a built-in tech/skills list, "
    "they don't replace it.",
    value="",
)

run_button = st.button("Find My Top 10 Projects 🚀")

if run_button:
    if not cv_file:
        st.warning("Please upload your CV to begin.")
        st.stop()

    if not os.path.exists(PROJECTS_CSV):
        st.error(f"'{PROJECTS_CSV}' not found. Please run the scraper first.")
        st.stop()

    mtime = os.path.getmtime(PROJECTS_CSV)
    df, status_msg = load_projects(mtime)
    if df is None:
        st.error(status_msg)
        st.stop()

    with st.spinner("Reading your CV..."):
        cv_text = extract_cv_text(cv_file)
        if not cv_text.strip():
            st.error("Could not extract any text from this PDF. Try a different export/version.")
            st.stop()
        cv_chunks = chunk_cv_text(cv_text)

    with st.spinner(f"Embedding your CV ({len(cv_chunks)} sections) and all projects..."):
        bi_encoder = load_bi_encoder()
        cv_chunk_embeddings = bi_encoder.encode(cv_chunks, show_progress_bar=False)

        field_texts_by_column = {col: df[col].tolist() for col in FIELD_COLUMNS}
        project_field_embeddings = embed_project_fields(mtime, field_texts_by_column)

    with st.spinner("Scoring all projects..."):
        extra_keywords = [k for k in user_keywords_input.split(",") if k.strip()]

        n_projects = len(df)
        field_scores = {col: np.zeros(n_projects) for col in FIELD_COLUMNS}

        for col in FIELD_COLUMNS:
            proj_embs = project_field_embeddings[col]  # (n_projects, dim)
            # For each project field, take the BEST matching CV chunk rather
            # than one truncated whole-CV embedding. This is what lets the
            # model "see" your whole CV instead of just the first page.
            sims = cosine_similarity(cv_chunk_embeddings, proj_embs)  # (n_chunks, n_projects)
            field_scores[col] = sims.max(axis=0)

        semantic_score = np.zeros(n_projects)
        for col in FIELD_COLUMNS:
            semantic_score += FIELD_WEIGHTS[col] * field_scores[col]

        keyword_scores = np.zeros(n_projects)
        matched_keywords_list = []
        for i in range(n_projects):
            score, matched = extract_matched_keywords(cv_text, df.iloc[i]["Combined_Text"], extra_keywords)
            keyword_scores[i] = score
            matched_keywords_list.append(matched)

        combined_score = SEMANTIC_WEIGHT * semantic_score + KEYWORD_WEIGHT * keyword_scores

        df = df.copy()
        df["Semantic_Score"] = np.round(semantic_score * 100, 2)
        df["Keyword_Score"] = np.round(keyword_scores * 100, 2)
        df["Combined_Score"] = np.round(combined_score * 100, 2)
        df["Matched_Keywords"] = [", ".join(sorted(m)) if m else "" for m in matched_keywords_list]
        for col in FIELD_COLUMNS:
            df[f"{col}_Score"] = np.round(field_scores[col] * 100, 2)

    with st.spinner("Reranking top candidates for precision..."):
        pool = df.sort_values(by="Combined_Score", ascending=False).head(
            min(RERANK_POOL_SIZE, n_projects)
        ).copy()

        cross_encoder = load_cross_encoder()
        # Pick whichever CV chunk is, on average, most relevant across all
        # projects and fields - that's the chunk we hand to the reranker as
        # "the CV" for a direct, context-aware comparison (rather than
        # comparing pre-computed vectors, which is what the bi-encoder does).
        total_relevance_per_chunk = sum(
            FIELD_WEIGHTS[c] * cosine_similarity(cv_chunk_embeddings, project_field_embeddings[c])
            for c in FIELD_COLUMNS
        ).sum(axis=1)
        best_chunk_idx = int(np.argmax(total_relevance_per_chunk))
        best_cv_text = cv_chunks[best_chunk_idx] if cv_chunks else cv_text[:1000]

        pairs = [[best_cv_text, row["Combined_Text"]] for _, row in pool.iterrows()]
        rerank_scores = cross_encoder.predict(pairs)

        # Normalize reranker scores to 0-1 for blending (raw logits can be
        # any range depending on the model).
        rr = np.array(rerank_scores, dtype=float)
        if rr.max() != rr.min():
            rr_norm = (rr - rr.min()) / (rr.max() - rr.min())
        else:
            rr_norm = np.zeros_like(rr)

        pool["Rerank_Score"] = np.round(rr_norm * 100, 2)
        pool["Final_Score"] = np.round(
            RERANK_WEIGHT * rr_norm * 100 + BIENCODER_WEIGHT * pool["Combined_Score"], 2
        )

        top_10 = pool.sort_values(by="Final_Score", ascending=False).head(10)

    st.success("Analysis complete!")
    st.markdown("### 🏆 Your Top 10 Project Matches")
    st.caption(
        "Final_Score blends a precision reranker with the weighted field "
        "match. Scores are relative to each other, not an absolute percentage "
        "- focus on rank order and the field/keyword breakdown, not the raw number."
    )

    display_columns = [
        "Final_Score", "Project_ID", "Title", "Faculty_Supervisor",
        "Project_Location", "Language", "Preferred_Start_Date",
        "Matched_Keywords",
        "Required_Skills_Score", "Student_Roles_Score",
        "Project_Activities_Score", "Project_Description_Score",
    ]
    existing_columns = [c for c in display_columns if c in top_10.columns]

    st.dataframe(
        top_10[existing_columns],
        hide_index=True,
        use_container_width=True,
    )

    with st.expander("🔍 See full project text for each top match"):
        for _, row in top_10.iterrows():
            st.markdown(f"**{row['Title']}** (Project {row['Project_ID']}, Final Score {row['Final_Score']})")
            for col in FIELD_COLUMNS:
                if str(row.get(col, "")).strip():
                    st.markdown(f"- **{col.replace('_', ' ')}**: {row[col]}")
            st.markdown("---")
