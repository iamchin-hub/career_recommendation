"""Evidence page for the Hakbang PH multipage Streamlit app."""

from __future__ import annotations

import streamlit as st

from career_engine import EVIDENCE_CHECKED, SOURCES


APP_BUILD_ID = "2026.07.28-multipage-v8"

MARKET_AND_AI_KEYS = [
    "psa_lfs",
    "dole_forecast",
    "tesda_5ir",
    "wef_jobs",
    "wef_skills",
    "wef_industry",
    "ilo_genai",
]
OCCUPATION_KEYS = [key for key in SOURCES if key.startswith("onet_")]
STANDARD_AND_CREDENTIAL_KEYS = [
    key
    for key in SOURCES
    if key not in set(MARKET_AND_AI_KEYS + OCCUPATION_KEYS)
]


def render_source_group(title: str, description: str, source_keys: list[str]) -> None:
    """Render a source category directly from the shared evidence registry."""
    st.header(title)
    st.write(description)
    for source_key in source_keys:
        source = SOURCES[source_key]
        st.markdown(
            f"**[{source['name']}]({source['url']})**  \n"
            f"{source['owner']} · {source['published']}"
        )


st.html(
    """
    <section class="page-hero">
      <div class="eyebrow">Research and official-source library</div>
      <h1>Evidence</h1>
      <p>
        This page is separate from the career scan. Every item below comes from
        the same source registry used by the recommendation explanations.
      </p>
    </section>
    """
)

metric_columns = st.columns(3)
metric_columns[0].metric("Source records", len(SOURCES))
metric_columns[1].metric("Last reviewed", EVIDENCE_CHECKED)
metric_columns[2].metric("Runtime AI-generated claims", "0")

st.info(
    f"The registry was reviewed on {EVIDENCE_CHECKED}. The May 2026 Philippine "
    "Labor Force Survey is the latest released national LFS result available at "
    "that review date; PSA scheduled the June 2026 preliminary results for "
    "6 August 2026. The page should be reviewed again after that release."
)

st.header("How to interpret this library")
st.markdown(
    """
    - **Direct evidence** supports a statement about the named occupation,
      standard, credential, or report.
    - **Directional evidence** describes a wider labor-market or technology trend.
      It does not prove that a particular employer is hiring.
    - **Inference** is used when a broader source is applied to a job family. The
      app labels that application instead of presenting it as a measured vacancy.
    - This is a dated registry, not a live vacancy feed. Recheck current job
      postings and official provider pages before making a career or payment decision.
    """
)

render_source_group(
    "Philippine labor market and AI-era research",
    "National labor-market context and global evidence about jobs, skills, and "
    "generative-AI exposure.",
    MARKET_AND_AI_KEYS,
)
render_source_group(
    "Occupation and task references",
    "Official O*NET occupation profiles are used to explain common work "
    "activities and why particular skills are relevant. O*NET is a U.S. "
    "occupational reference, not a Philippine license or local vacancy count.",
    OCCUPATION_KEYS,
)
render_source_group(
    "Standards, certifications, and practitioner evidence",
    "Official standards bodies, regulators, and certification providers support "
    "credential and practice statements. Eligibility, price, language, and exam "
    "versions must be checked again on the provider's page.",
    STANDARD_AND_CREDENTIAL_KEYS,
)

with st.expander("Freshness and maintenance policy"):
    st.markdown(
        f"""
        1. Time-sensitive links are checked against the original publisher.
        2. `EVIDENCE_CHECKED` is changed only after that review is completed.
        3. The Evidence page renders `SOURCES` directly; it does not maintain a
           second copy of names, links, owners, or publication dates.
        4. Recommendation explanations reference source keys from the same registry.
        5. Newer official releases should replace or supplement older evidence only
           after their scope and limitations are checked.

        Current registry review date: **{EVIDENCE_CHECKED}**.
        """
    )

st.caption(
    f"App build: {APP_BUILD_ID} · Evidence reviewed: {EVIDENCE_CHECKED}. "
    "Recheck official credential pages before paying or enrolling."
)
