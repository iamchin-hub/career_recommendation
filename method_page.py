"""Method page for the Hakbang PH multipage Streamlit app."""

from __future__ import annotations

from io import StringIO

import streamlit as st

from career_engine import (
    CAREERS,
    DATASET_VERSION,
    EVIDENCE_CHECKED,
    MODEL_NAME,
    RANKING_WEIGHTS,
    SKILLS,
    SOURCES,
    SUPPORT_THRESHOLDS,
    SYNTHETIC_CV_ACCURACY,
    SYNTHETIC_CV_MACRO_F1,
    SYNTHETIC_PROFILES_PER_CAREER,
    CareerEngine,
)


APP_BUILD_ID = "2026.07.28-multipage-v8"

WEIGHT_LABELS = {
    "skill_alignment": "Direct skill alignment",
    "core_skill_coverage": "Core-skill coverage",
    "synthetic_model_fit": "Synthetic model fit",
    "experience_proximity": "Experience proximity",
    "current_demand_evidence": "Current-demand evidence grade",
    "future_demand_evidence": "Future-demand evidence grade",
}


@st.cache_resource(show_spinner="Preparing the teaching model…")
def load_method_engine(cache_key: str) -> CareerEngine:
    """Use the same deterministic training procedure as the career-scan page."""
    del cache_key
    return CareerEngine.train()


engine = load_method_engine(f"{DATASET_VERSION}:{APP_BUILD_ID}")

st.html(
    f"""
    <section class="page-hero">
      <div class="eyebrow">How the recommendation is produced</div>
      <h1>Method</h1>
      <p>
        This page is separate from the career scan. Its numbers are read directly
        from the model constants used by the recommender, so the explanation and
        the running code stay synchronized.
      </p>
    </section>
    """
)

metric_columns = st.columns(4)
metric_columns[0].metric("Synthetic profiles", f"{len(engine.synthetic_profiles):,}")
metric_columns[1].metric("Job families", len(CAREERS))
metric_columns[2].metric("Skill signals", len(SKILLS))
metric_columns[3].metric("Evidence last reviewed", EVIDENCE_CHECKED)

st.header("What the model uses")
st.markdown(
    f"""
    The classifier is **{MODEL_NAME}**. It is trained on
    **{len(engine.synthetic_profiles):,} deterministic synthetic profiles**:
    {SYNTHETIC_PROFILES_PER_CAREER} profiles for each of {len(CAREERS)} job
    families.

    The only input features are:

    - total years of work experience; and
    - the {len(SKILLS)} self-rated transferable skills, each from 0 to 5.

    Industry, employer, current job title, education, age, sex, ethnicity,
    disability, and other protected characteristics are not model features.
    """
)

st.header("How jobs are ranked")
st.write(
    "The classifier is only one part of the ranking. Most of the score comes from "
    "transparent comparison of the professional's demonstrated skills with the "
    "documented skill pattern for each job family."
)
for key, weight in RANKING_WEIGHTS.items():
    st.markdown(f"- **{WEIGHT_LABELS[key]}: {weight:.0%}**")

st.caption(
    f"These values total {sum(RANKING_WEIGHTS.values()):.0%} and are read from "
    "the same RANKING_WEIGHTS object used in career_engine.py."
)

st.subheader("Minimum support rules")
st.markdown(
    f"""
    A job is shown only when all three conditions are met:

    - skill alignment is at least
      **{SUPPORT_THRESHOLDS['skill_alignment']:.0f}/100**;
    - core-skill coverage is at least
      **{SUPPORT_THRESHOLDS['core_skill_coverage']:.0f}/100**; and
    - the combined recommendation score is at least
      **{SUPPORT_THRESHOLDS['recommendation_score']:.0f}/100**.

    If fewer than three jobs pass, the app returns fewer than three. If none
    pass, it abstains rather than filling the result with unrelated jobs.
    A zero skill rating is valid and means no demonstrated experience.
    """
)

st.header("What the benchmark means")
st.info(
    f"In seeded five-fold testing, {MODEL_NAME} had mean macro-F1 "
    f"{SYNTHETIC_CV_MACRO_F1:.3f} and mean accuracy "
    f"{SYNTHETIC_CV_ACCURACY:.3f}. These figures measure how well the model "
    "recovers synthetic labeling rules. They do not measure real hiring success, "
    "career success, salary, or vacancy probability."
)

st.header("Where factual explanations come from")
st.markdown(
    f"""
    Career tasks, demand context, AI-era explanations, credentials, and courses
    come from a fixed registry containing **{len(SOURCES)} source records**.
    Runtime text is selected from reviewed content; the classifier does not invent
    factual claims. The registry's current review date is
    **{EVIDENCE_CHECKED}**.
    """
)

st.header("Download the teaching dataset")
csv_buffer = StringIO()
engine.synthetic_profiles.to_csv(csv_buffer, index=False)
st.download_button(
    f"Download {len(engine.synthetic_profiles):,} synthetic profiles (.csv)",
    data=csv_buffer.getvalue(),
    file_name="hakbang_ph_2200_skills_first_profiles.csv",
    mime="text/csv",
)
st.caption(
    f"Dataset/model version: {DATASET_VERSION}. It contains no real people, "
    "résumés, employers, vacancies, or employment outcomes."
)

with st.expander("Important limitations"):
    st.markdown(
        """
        - Synthetic rows add consistency for teaching; they do not add real-world truth.
        - Self-assessed skills may be incomplete or inconsistent.
        - Demand grades are editorial evidence mappings, not live vacancy counts.
        - Global research does not automatically represent every Philippine region.
        - The catalog contains eleven job families and omits many valid careers.
        - The model has not been validated with real applicants or career outcomes.
        - Years of experience do not prove domain depth, leadership, or seniority.
        - Certification requirements, prices, languages, and exam versions can change.
        """
    )

st.caption(
    f"App build: {APP_BUILD_ID} · Dataset: {DATASET_VERSION} · "
    f"Evidence reviewed: {EVIDENCE_CHECKED}"
)
