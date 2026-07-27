"""Public Streamlit interface for the Hakbang PH career recommender."""

from __future__ import annotations

from io import StringIO

import streamlit as st

from career_engine import (
    CAREER_GOALS,
    DATASET_VERSION,
    EVIDENCE_CHECKED,
    INDUSTRIES,
    SKILLS,
    SOURCES,
    CareerEngine,
)


st.set_page_config(
    page_title="Hakbang PH · Career Move Explorer",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      :root {
        --ink: #17322e;
        --green: #0b6e69;
        --green-dark: #07514d;
        --mint: #eaf6f2;
        --cream: #fffaf0;
        --gold: #e5983e;
      }
      .stApp {
        background:
          radial-gradient(circle at 90% 4%, rgba(229,152,62,.13), transparent 24rem),
          linear-gradient(180deg, #fbfdfb 0%, #f5faf7 48%, #fffaf3 100%);
        color: var(--ink);
      }
      .block-container {max-width: 1120px; padding-top: 2rem; padding-bottom: 4rem;}
      .hero {
        padding: 2.2rem 2.3rem;
        border-radius: 28px;
        background: linear-gradient(125deg, #073f3b 0%, #0b6e69 68%, #16928a 100%);
        box-shadow: 0 20px 55px rgba(7,63,59,.16);
        color: white;
        margin-bottom: 1.2rem;
      }
      .hero-kicker {
        color: #ffd799;
        font-size: .82rem;
        font-weight: 750;
        letter-spacing: .12em;
        text-transform: uppercase;
      }
      .hero h1 {
        color: white;
        font-size: clamp(2rem, 5vw, 4rem);
        line-height: 1.02;
        margin: .55rem 0 .8rem;
        letter-spacing: -.04em;
      }
      .hero p {font-size: 1.06rem; max-width: 740px; color: #e6f7f2; margin: 0;}
      .truth-note {
        border: 1px solid #f1cd8f;
        background: #fff8e7;
        border-radius: 16px;
        padding: 1rem 1.15rem;
        color: #634619;
        margin: .7rem 0 1.5rem;
      }
      .eyebrow {
        color: var(--green);
        font-weight: 800;
        letter-spacing: .08em;
        text-transform: uppercase;
        font-size: .76rem;
      }
      .role-head {
        border-left: 6px solid var(--gold);
        background: white;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        margin: .25rem 0 1rem;
        box-shadow: 0 8px 24px rgba(23,50,46,.06);
      }
      .role-head h3 {margin: .15rem 0 .35rem;}
      .role-head p {margin: 0; color: #4b625e;}
      div[data-testid="stMetric"] {
        background: rgba(255,255,255,.88);
        border: 1px solid #dcece7;
        padding: .8rem;
        border-radius: 14px;
      }
      div[data-testid="stForm"] {
        background: rgba(255,255,255,.84);
        border: 1px solid #dbece6;
        border-radius: 22px;
        padding: 1.2rem 1.3rem;
      }
      .stButton > button, .stFormSubmitButton > button {
        border-radius: 999px;
        font-weight: 750;
      }
      a {color: var(--green-dark);}
      footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Training the 100-profile synthetic model…")
def load_engine() -> CareerEngine:
    return CareerEngine.train()


def source_links(source_keys: list[str]) -> None:
    for source_key in source_keys:
        source = SOURCES[source_key]
        st.markdown(
            f"- [{source['name']}]({source['url']}) — "
            f"{source['owner']}, {source['published']}"
        )


def render_recommendation(result: dict, rank: int) -> None:
    st.markdown(
        f"""
        <div class="role-head">
          <span class="eyebrow">Recommendation {rank}</span>
          <h3>{result["career"]}</h3>
          <p>{result["summary"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    score_col, model_col, skill_col, demand_col = st.columns(4)
    score_col.metric("Comparative score", f"{result['recommendation_score']:.1f}/100")
    model_col.metric("Synthetic model fit", f"{result['synthetic_model_fit']:.1f}%")
    skill_col.metric("Skill fit index", f"{result['skill_fit']:.1f}%")
    demand_col.metric(
        "Demand direction",
        result["future_demand"]["label"],
        f"Current: {result['current_demand']['label']}",
    )
    st.caption(
        "These are ranking indices—not probabilities of getting hired, succeeding, "
        "or earning a particular salary."
    )

    overview_tab, ai_tab, skills_tab, credential_tab = st.tabs(
        ["Why this move", "AI-era opportunity", "Skill-building", "Certification"]
    )

    with overview_tab:
        current_col, future_col = st.columns(2)
        with current_col:
            st.subheader("Current demand")
            st.markdown(
                f"**{result['current_demand']['label']} · "
                f"{result['current_demand']['basis']}**"
            )
            st.write(result["current_demand"]["insight"])
            with st.expander("Current-demand sources"):
                source_links(result["current_demand"]["sources"])
        with future_col:
            st.subheader("Future demand")
            st.markdown(
                f"**{result['future_demand']['label']} · "
                f"{result['future_demand']['basis']}**"
            )
            st.write(result["future_demand"]["insight"])
            with st.expander("Future-demand sources"):
                source_links(result["future_demand"]["sources"])

    with ai_tab:
        st.subheader("How AI can expand the role")
        st.write(result["ai_opportunity"])
        st.subheader("The human advantage")
        st.write(result["human_edge"])
        st.subheader("A practical portfolio proof")
        st.info(result["first_proof"])

    with skills_tab:
        st.subheader("Priority gaps against the demo target")
        if result["skill_gaps"]:
            for gap in result["skill_gaps"]:
                st.markdown(
                    f"- **{gap['label']}** — your rating "
                    f"{gap['current']:g}/5; demo target {gap['target']}/5"
                )
        else:
            st.success(
                "Your self-ratings meet this synthetic prototype's target levels. "
                "Validate them next with a work sample and practitioner feedback."
            )
        st.caption(
            "Targets come from documented synthetic role prototypes. They are not "
            "employer requirements or occupational standards."
        )

    with credential_tab:
        credential = result["certification"]
        st.subheader(credential["name"])
        st.markdown(f"**Official provider:** {credential['issuer']}")
        st.write(credential["why_it_fits"])
        st.markdown(f"**Eligibility note:** {credential['eligibility']}")
        st.link_button(
            "Open the official certification page ↗",
            credential["url"],
            use_container_width=True,
        )
        st.divider()
        st.markdown(f"**Practitioner evidence:** {credential['practitioner']}")
        st.caption(credential["source_type"])
        st.write(credential["practitioner_insight"])
        st.markdown(
            f"[Read the original account or survey ↗]"
            f"({credential['practitioner_url']})"
        )
        st.warning(credential["caveat"])


st.markdown(
    """
    <section class="hero">
      <div class="hero-kicker">Hakbang PH · Career Move Explorer</div>
      <h1>Find your next move in an AI-shaped workplace.</h1>
      <p>
        Compare three career pathways using your experience, self-assessed
        skills, current industry, and a dated research ledger—then leave with a
        practical project and an official credential to investigate.
      </p>
    </section>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="truth-note">
      <strong>Research prototype:</strong> the statistical model learns from
      exactly 100 synthetic profiles—not real career outcomes. Results support
      exploration and should not be used for hiring, promotion, redundancy,
      compensation, or another high-impact decision.
    </div>
    """,
    unsafe_allow_html=True,
)

input_tab, method_tab, evidence_tab = st.tabs(
    ["Build my profile", "How it works", "Evidence library"]
)

engine = load_engine()

with input_tab:
    st.header("Tell us what you can demonstrate today")
    st.write(
        "Use the 1–5 skill scale honestly: 1 means new to the skill; 5 means you "
        "can independently deliver strong work and explain your decisions."
    )
    with st.form("career_profile"):
        context_col, experience_col, goal_col = st.columns(3)
        with context_col:
            industry = st.selectbox(
                "Current industry",
                options=list(INDUSTRIES),
                format_func=INDUSTRIES.get,
            )
        with experience_col:
            years_experience = st.number_input(
                "Total years of work experience",
                min_value=0.0,
                max_value=50.0,
                value=5.0,
                step=0.5,
            )
            leadership_years = st.number_input(
                "Years leading people or major work",
                min_value=0.0,
                max_value=50.0,
                value=1.0,
                step=0.5,
            )
        with goal_col:
            goal = st.selectbox(
                "Main career goal",
                options=list(CAREER_GOALS),
                format_func=CAREER_GOALS.get,
            )

        st.subheader("Self-assess your transferable skills")
        skill_columns = st.columns(3)
        skill_values: dict[str, int] = {}
        for index, (skill_key, skill) in enumerate(SKILLS.items()):
            with skill_columns[index % 3]:
                skill_values[skill_key] = st.slider(
                    skill["label"],
                    min_value=1,
                    max_value=5,
                    value=3,
                    help=skill["hint"],
                    key=f"skill_{skill_key}",
                )

        submitted = st.form_submit_button(
            "Show my three career moves",
            type="primary",
            use_container_width=True,
        )

    st.caption(
        "Privacy: this demo does not ask for your name, age, employer, résumé, or "
        "protected characteristics. Inputs remain only in the current app session."
    )

    if submitted:
        profile = {
            "current_industry": industry,
            "years_experience": years_experience,
            "leadership_years": leadership_years,
            "goal": goal,
            **skill_values,
        }
        try:
            st.session_state["recommendations"] = engine.recommend(profile)
        except ValueError as error:
            st.error(str(error))

    if "recommendations" in st.session_state:
        st.divider()
        st.header("Your three next-move hypotheses")
        st.write(
            "Treat these as options to investigate through conversations, work "
            "samples, and current job postings—not as instructions."
        )
        recommendation_tabs = st.tabs(
            [
                f"{rank}. {item['career']}"
                for rank, item in enumerate(
                    st.session_state["recommendations"],
                    start=1,
                )
            ]
        )
        for rank, (tab, result) in enumerate(
            zip(recommendation_tabs, st.session_state["recommendations"]),
            start=1,
        ):
            with tab:
                render_recommendation(result, rank)

with method_tab:
    st.header("A transparent two-stage recommender")
    st.markdown(
        """
        1. **Synthetic model fit:** a logistic-regression classifier learns
           patterns from 100 seeded synthetic profiles—ten per career pathway.
        2. **Evidence reranking:** the app combines model fit with direct skill
           fit, experience, industry adjacency, and dated demand grades.

        The reranking weights are:

        - synthetic model fit: **38%**
        - direct skill fit: **24%**
        - experience fit: **8%**
        - current-demand evidence grade: **8%**
        - future-demand evidence grade: **12%**
        - industry adjacency: **6%**
        - goal adjustment: **up to 4%**
        """
    )
    st.info(
        "In the validated notebook run dated 27 July 2026, logistic regression "
        "recorded mean macro-F1 0.876 and mean accuracy 0.880 in seeded "
        "stratified five-fold cross-validation. These are synthetic-rule recovery "
        "scores—not real-world career accuracy."
    )
    st.subheader("Download the teaching dataset")
    csv_buffer = StringIO()
    engine.synthetic_profiles.to_csv(csv_buffer, index=False)
    st.download_button(
        "Download 100 synthetic profiles (.csv)",
        data=csv_buffer.getvalue(),
        file_name="hakbang_ph_100_synthetic_profiles.csv",
        mime="text/csv",
    )
    st.caption(
        f"Dataset/model version: {DATASET_VERSION}. No real people, résumés, "
        "employers, vacancies, or employment outcomes are included."
    )
    with st.expander("Important limitations"):
        st.markdown(
            """
            - A 100-row synthetic dataset cannot validate real-world career outcomes.
            - Self-assessed skills can be inconsistent.
            - Demand grades are ordinal editorial mappings, not live job counts.
            - Global evidence does not automatically describe every Philippine region.
            - The catalogue contains only ten pathways and omits many valid careers.
            - Practitioner accounts may contain selection and survivorship bias.
            - Certification requirements, prices, and exam versions can change.
            """
        )

with evidence_tab:
    st.header("Fixed research sources")
    st.write(
        "The app does not generate factual claims with an LLM. Career evidence "
        "is selected from this fixed registry, with inference labeled explicitly."
    )
    for source in SOURCES.values():
        st.markdown(
            f"**[{source['name']}]({source['url']})**  \n"
            f"{source['owner']} · {source['published']}"
        )
    st.caption(
        f"Evidence and official credential links were last reviewed "
        f"{EVIDENCE_CHECKED}. Recheck the provider page before paying or enrolling."
    )

