"""Public Streamlit interface for the Hakbang PH career recommender."""

from __future__ import annotations

from io import StringIO

import streamlit as st

from career_engine import (
    DATASET_VERSION,
    EVIDENCE_CHECKED,
    MODEL_NAME,
    SKILLS,
    SOURCES,
    SYNTHETIC_CV_ACCURACY,
    SYNTHETIC_CV_MACRO_F1,
    SYNTHETIC_PROFILES_PER_CAREER,
    CareerEngine,
)

APP_BUILD_ID = "2026.07.28-zero-valid-v7"
ENGINE_CACHE_KEY = f"{DATASET_VERSION}:{APP_BUILD_ID}"


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
        --night: #030713;
        --panel: #111827;
        --panel-soft: #182235;
        --line: rgba(151, 172, 214, .22);
        --text: #f7f8fb;
        --muted: #ffffff;
        --blue: #52a8ff;
        --blue-deep: #246bfe;
        --orange: #ff6b2c;
        --gold: #ffbd35;
      }
      .stApp {
        background:
          radial-gradient(circle at 78% 3%, rgba(34, 107, 254, .35), transparent 27rem),
          radial-gradient(circle at 10% 40%, rgba(12, 49, 118, .24), transparent 30rem),
          linear-gradient(180deg, #030713 0%, #071022 54%, #050914 100%);
        color: var(--text);
      }
      .stApp,
      .stApp [data-testid="stMarkdownContainer"],
      .stApp [data-testid="stMarkdownContainer"] p,
      .stApp [data-testid="stMarkdownContainer"] li,
      .stApp [data-testid="stCaptionContainer"],
      .stApp [data-testid="stCaptionContainer"] p,
      .stApp [data-testid="stWidgetLabel"],
      .stApp [data-testid="stWidgetLabel"] p,
      .stApp [data-testid="stMetricLabel"],
      .stApp [data-testid="stMetricLabel"] p,
      .stApp [data-testid="stMetricValue"],
      .stApp [data-testid="stMetricValue"] div,
      .stApp [data-testid="stSlider"] p,
      .stApp [data-testid="stSlider"] span {
        color: #ffffff !important;
        opacity: 1 !important;
      }
      .stApp input,
      .stApp textarea,
      .stApp [data-baseweb="select"] *,
      .stApp [data-testid="stNumberInput"] button {
        color: #101827 !important;
      }
      [role="listbox"],
      [role="listbox"] [role="option"],
      [role="listbox"] [role="option"] *,
      [data-baseweb="popover"] [role="option"],
      [data-baseweb="popover"] [role="option"] * {
        color: #101827 !important;
      }
      .stApp input::placeholder,
      .stApp textarea::placeholder {
        color: #596579 !important;
        opacity: 1 !important;
      }
      .stApp [data-testid="stTooltipIcon"] button,
      .stApp [data-testid="stTooltipIcon"] svg {
        color: #ffffff !important;
        stroke: currentColor !important;
        opacity: 1 !important;
      }
      [data-testid="stHeader"] {background: transparent;}
      [data-testid="stToolbar"] {right: 1rem;}
      .block-container {
        max-width: 1180px;
        padding-top: 1.15rem;
        padding-bottom: 5rem;
      }
      .topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: .7rem .85rem .7rem 1rem;
        border: 1px solid var(--line);
        border-radius: 14px;
        background: rgba(15, 22, 37, .76);
        backdrop-filter: blur(18px);
        margin-bottom: 1rem;
      }
      .brand {
        display: flex;
        align-items: center;
        gap: .7rem;
        color: white;
        font-weight: 850;
        letter-spacing: -.02em;
      }
      .brand-mark {
        display: inline-grid;
        place-items: center;
        width: 31px;
        height: 31px;
        border-radius: 9px;
        color: #04101f;
        background: linear-gradient(135deg, #fff 0%, #9fd0ff 100%);
        box-shadow: 0 0 30px rgba(82, 168, 255, .42);
      }
      .nav-pills {
        display: flex;
        align-items: center;
        gap: .45rem;
      }
      .nav-pills button {
        appearance: none;
        border: 0;
        background: transparent;
        cursor: pointer;
        font: inherit;
        color: #d8e3f4 !important;
        padding: .48rem .78rem;
        border-radius: 9px;
        font-size: .83rem;
      }
      .nav-pills button:hover {background: rgba(255,255,255,.07);}
      .nav-pills .nav-cta {
        border: 1px solid rgba(255,255,255,.62);
        color: white !important;
        padding-inline: 1rem;
      }
      .hero {
        position: relative;
        display: grid;
        grid-template-columns: minmax(0, 1.04fr) minmax(320px, .96fr);
        min-height: 500px;
        overflow: hidden;
        border: 1px solid var(--line);
        border-radius: 28px 28px 0 0;
        background:
          radial-gradient(circle at 76% 35%, rgba(45, 120, 255, .58), transparent 28%),
          radial-gradient(circle at 82% 58%, rgba(17, 75, 181, .48), transparent 36%),
          linear-gradient(120deg, #02050d 0%, #06132d 58%, #102c66 100%);
        box-shadow: 0 30px 90px rgba(0,0,0,.36);
      }
      .hero-copy {
        z-index: 3;
        display: flex;
        flex-direction: column;
        justify-content: center;
        padding: 3.4rem 0 3rem 3.2rem;
      }
      .hero-kicker {
        color: #8fc9ff;
        font-size: .82rem;
        font-weight: 750;
        letter-spacing: .12em;
        text-transform: uppercase;
      }
      .hero h1 {
        color: white;
        font-size: clamp(3.2rem, 5vw, 4.6rem);
        line-height: .96;
        max-width: 690px;
        margin: .8rem 0 1.2rem;
        letter-spacing: -.055em;
      }
      .hero p {
        font-size: 1rem;
        line-height: 1.65;
        max-width: 630px;
        color: #c6d3e7;
        margin: 0;
      }
      .hero-actions {
        display: flex;
        flex-wrap: wrap;
        gap: .7rem;
        margin-top: 1.8rem;
      }
      .hero-actions button {
        appearance: none;
        cursor: pointer;
        font: inherit;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        text-decoration: none;
        font-weight: 800;
        border-radius: 10px;
        padding: .78rem 1.1rem;
      }
      .hero-primary {
        color: #06101e !important;
        background: linear-gradient(135deg, #6eb8ff, #3f91ff);
        box-shadow: 0 12px 35px rgba(63, 145, 255, .28);
      }
      .hero-secondary {
        color: #eef5ff !important;
        border: 1px solid var(--line);
        background: rgba(255,255,255,.045);
      }
      .hero-visual {
        position: relative;
        min-height: 500px;
      }
      .machine-glow {
        position: absolute;
        inset: 10% 3% 3% 2%;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(45,128,255,.26), transparent 67%);
        filter: blur(3px);
      }
      .orbit {
        position: absolute;
        border: 2px solid rgba(116, 177, 255, .28);
        border-radius: 50%;
        box-shadow: inset 0 0 22px rgba(63, 145, 255, .14);
      }
      .orbit-a {width: 360px; height: 360px; top: 92px; right: 64px; transform: rotate(-17deg);}
      .orbit-b {width: 265px; height: 420px; top: 55px; right: 110px; transform: rotate(39deg); border-color: rgba(255,107,44,.34);}
      .orbit-c {width: 170px; height: 170px; top: 190px; right: 157px; border-color: rgba(255,189,53,.38);}
      .machine-core {
        position: absolute;
        top: 230px;
        right: 194px;
        display: grid;
        place-items: center;
        width: 94px;
        height: 94px;
        border: 9px solid #102b69;
        border-radius: 50%;
        color: white;
        font-weight: 900;
        letter-spacing: -.04em;
        background: radial-gradient(circle at 35% 30%, #74c0ff, #1d63f0 48%, #071634 72%);
        box-shadow:
          0 0 0 12px rgba(0,0,0,.32),
          0 0 70px rgba(55,137,255,.62);
      }
      .track {
        position: absolute;
        height: 24px;
        border-radius: 999px;
        box-shadow: 0 14px 25px rgba(0,0,0,.3);
      }
      .track-a {
        width: 235px;
        top: 160px;
        right: 25px;
        background: linear-gradient(90deg, #ff4826, #ff8a22, #ffd446);
        transform: rotate(12deg);
      }
      .track-b {
        width: 260px;
        top: 370px;
        right: 34px;
        background: linear-gradient(90deg, #1f5af1, #65b4ff);
        transform: rotate(-20deg);
      }
      .track-c {
        width: 215px;
        top: 306px;
        right: 225px;
        background: linear-gradient(90deg, #ffbd35, #ff5d28);
        transform: rotate(31deg);
      }
      .signal-card {
        position: absolute;
        z-index: 4;
        padding: .68rem .82rem;
        border: 1px solid rgba(255,255,255,.28);
        border-radius: 12px;
        color: white;
        font-size: .82rem;
        font-weight: 760;
        backdrop-filter: blur(10px);
        box-shadow: 0 15px 35px rgba(0,0,0,.28);
      }
      .signal-one {top: 87px; right: 30px; background: rgba(54,132,255,.88);}
      .signal-two {top: 405px; right: 265px; background: rgba(255,100,40,.85);}
      .signal-three {top: 106px; right: 260px; background: rgba(14,23,43,.86);}
      .proof-strip {
        display: grid;
        grid-template-columns: 1.25fr repeat(3, 1fr);
        gap: .8rem;
        align-items: stretch;
        padding: 1.25rem;
        border: 1px solid var(--line);
        border-top: 0;
        border-radius: 0 0 28px 28px;
        background: linear-gradient(135deg, #151e2b, #101722);
        margin-bottom: 1.3rem;
      }
      .proof-quote {
        padding: .8rem 1rem;
      }
      .proof-quote strong {
        display: block;
        font-size: 1.25rem;
        margin-bottom: .25rem;
      }
      .proof-quote span {color: var(--muted); font-size: .84rem;}
      .proof-chip {
        display: grid;
        place-items: center;
        min-height: 76px;
        padding: .75rem;
        text-align: center;
        border: 1px solid var(--line);
        border-radius: 14px;
        color: #dce8fa;
        background: rgba(255,255,255,.025);
        font-weight: 750;
      }
      .truth-note {
        border: 1px solid rgba(255,189,53,.32);
        background: rgba(102,68,12,.24);
        border-radius: 16px;
        padding: 1rem 1.15rem;
        color: #ffe0a0;
        margin: .7rem 0 1.5rem;
      }
      #career-scan,
      #how-it-works,
      #evidence-library {
        scroll-margin-top: 1.25rem;
      }
      .eyebrow {
        color: #72b9ff;
        font-weight: 800;
        letter-spacing: .08em;
        text-transform: uppercase;
        font-size: .76rem;
      }
      .role-head {
        border-left: 6px solid var(--gold);
        border-top: 1px solid var(--line);
        border-right: 1px solid var(--line);
        border-bottom: 1px solid var(--line);
        background: linear-gradient(135deg, rgba(25,37,58,.96), rgba(12,20,36,.96));
        border-radius: 16px;
        padding: 1rem 1.1rem;
        margin: .25rem 0 1rem;
        box-shadow: 0 16px 34px rgba(0,0,0,.2);
      }
      .role-head h3 {margin: .15rem 0 .35rem;}
      .role-head p {margin: 0; color: #ffffff;}
      div[data-testid="stMetric"] {
        background: rgba(17, 27, 45, .9);
        border: 1px solid var(--line);
        padding: .8rem;
        border-radius: 14px;
      }
      div[data-testid="stForm"] {
        background: linear-gradient(145deg, rgba(18,29,48,.95), rgba(9,16,30,.96));
        border: 1px solid var(--line);
        border-radius: 22px;
        padding: 1.35rem 1.45rem;
      }
      .stButton > button, .stFormSubmitButton > button {
        border-radius: 10px;
        font-weight: 750;
      }
      .stApp [data-testid="stLinkButton"] a,
      .stApp [data-testid="stLinkButton"] a *,
      .stApp .stLinkButton a,
      .stApp .stLinkButton a * {
        color: #050914 !important;
        fill: currentColor !important;
        opacity: 1 !important;
      }
      div[role="tablist"] {
        gap: .65rem;
        background: rgba(11,18,32,.72);
        border: 1px solid var(--line);
        border-radius: 13px;
        padding: .45rem .6rem;
        overflow-x: auto;
        overflow-y: hidden;
        scrollbar-width: thin;
      }
      [data-testid="stTab"] {
        flex: 0 0 auto !important;
        min-width: max-content !important;
        padding: .55rem 1rem !important;
        border-radius: 9px;
        color: #ffffff !important;
        background-color: transparent !important;
        white-space: nowrap !important;
        transition: background-color .16s ease, color .16s ease;
      }
      .stApp [data-testid="stTab"] p,
      .stApp [data-testid="stTab"] span {
        color: inherit !important;
        opacity: 1 !important;
      }
      .stApp [data-testid="stTab"][aria-selected="true"] {
        color: #ffffff !important;
        background-color: transparent !important;
      }
      .stApp [data-testid="stTab"][aria-selected="true"] p,
      .stApp [data-testid="stTab"][aria-selected="true"] span {
        color: #ffffff !important;
      }
      .stApp [data-testid="stTab"]:hover,
      .stApp [data-testid="stTab"][aria-selected="true"]:hover {
        color: #ffffff !important;
        background-color: #246bfe !important;
      }
      .stApp [data-testid="stTab"]:hover p,
      .stApp [data-testid="stTab"]:hover span,
      .stApp [data-testid="stTab"][aria-selected="true"]:hover p,
      .stApp [data-testid="stTab"][aria-selected="true"]:hover span {
        color: #ffffff !important;
      }
      [data-testid="stExpander"] details {
        overflow: hidden;
        border: 1px solid var(--line) !important;
        border-radius: 12px !important;
        background: rgba(17, 27, 45, .92) !important;
      }
      [data-testid="stExpander"] summary {
        color: #ffffff !important;
        background: rgba(17, 27, 45, .96) !important;
        transition: background-color .16s ease, color .16s ease;
      }
      [data-testid="stExpander"] summary p,
      [data-testid="stExpander"] summary span,
      [data-testid="stExpander"] summary svg {
        color: inherit !important;
        fill: currentColor !important;
        opacity: 1 !important;
      }
      [data-testid="stExpander"] details[open] > summary {
        color: #050914 !important;
        background: #ffffff !important;
      }
      .stApp [data-testid="stExpander"] details[open] > summary [data-testid="stMarkdownContainer"],
      .stApp [data-testid="stExpander"] details[open] > summary p,
      .stApp [data-testid="stExpander"] details[open] > summary span,
      .stApp [data-testid="stExpander"] details[open] > summary svg {
        color: #050914 !important;
        fill: currentColor !important;
      }
      [data-testid="stExpander"] summary:hover,
      [data-testid="stExpander"] details[open] > summary:hover {
        color: #ffffff !important;
        background: #246bfe !important;
      }
      .stApp [data-testid="stExpander"] details[open] > summary:hover [data-testid="stMarkdownContainer"],
      .stApp [data-testid="stExpander"] details[open] > summary:hover p,
      .stApp [data-testid="stExpander"] details[open] > summary:hover span,
      .stApp [data-testid="stExpander"] details[open] > summary:hover svg {
        color: #ffffff !important;
        fill: currentColor !important;
      }
      a {color: #77bbff;}
      footer {visibility: hidden;}
      @media (max-width: 820px) {
        .nav-pills a:not(.nav-cta) {display: none;}
        .hero {grid-template-columns: 1fr; min-height: auto;}
        .hero-copy {padding: 3.2rem 1.4rem 2rem;}
        .hero h1 {font-size: clamp(3rem, 15vw, 4.5rem);}
        .hero-visual {display: none;}
        .proof-strip {grid-template-columns: 1fr;}
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Training the 2,200-profile skills-first model…")
def load_engine(cache_key: str) -> CareerEngine:
    """Train once per explicit app/model build so deployments cannot reuse stale logic."""
    del cache_key
    return CareerEngine.train()


def source_links(source_keys: list[str]) -> None:
    for source_key in source_keys:
        source = SOURCES[source_key]
        st.markdown(
            f"- [{source['name']}]({source['url']}) — "
            f"{source['owner']}, {source['published']}"
        )


def render_recommendation(result: dict, rank: int) -> None:
    guidance = result["role_guidance"]
    st.markdown(
        f"""
        <div class="role-head">
          <span class="eyebrow">Skills-based match {rank}</span>
          <h3>{result["display_career"]}</h3>
          <p>{result["display_summary"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    score_col, second_col, third_col, role_col = st.columns(4)
    score_col.metric(
        "Comparative match",
        f"{result['recommendation_score']:.1f}/100",
    )
    second_col.metric(
        "Skill alignment",
        f"{result['skill_alignment']:.1f}%",
    )
    third_col.metric(
        "Core-skill coverage",
        f"{result['core_skill_coverage']:.1f}%",
    )
    role_col.metric(
        "AI-era relevance",
        f"{result['ai_competitiveness']:.1f}%",
    )
    st.caption(
        f"Supporting indices: synthetic model fit "
        f"{result['synthetic_model_fit']:.1f}% · experience proximity "
        f"{result['experience_fit']:.1f}%. These are comparative teaching "
        "indices—not hiring, success, salary, or job-availability probabilities."
    )

    overview_tab, ai_tab, skills_tab, learning_tab, evidence_tab = st.tabs(
        [
            "Why this job fits",
            "AI-era relevance",
            "Skills to build",
            "Certifications & courses",
            "Research evidence",
        ]
    )

    with overview_tab:
        st.subheader("The simple explanation")
        st.write(guidance["fit"])
        st.info(
            "Why the app showed this job: your strongest demonstrated skills overlap "
            "with important activities in this job family. This is a direction to "
            "investigate—not proof that you already meet an employer's requirements."
        )

        st.subheader("How your existing skills connect to the work")
        if result["matched_skills"]:
            for match in result["matched_skills"]:
                st.markdown(f"#### {match['label']}")
                st.caption(
                    f"Your rating: {match['current']:g}/5 · teaching target used "
                    f"for this job family: {match['target']}/5"
                )
                st.markdown(f"**Why it matters:** {match['why']}")
                st.markdown(f"**Example at work:** {match['example']}")
        else:
            st.write(
                "The app found a partial overall pattern, but none of your rated "
                "skills overlaps strongly with this role's four priority skills. "
                "Treat this option cautiously and validate it with a practitioner."
            )

        st.subheader("What people in this job commonly do")
        st.markdown(
            "\n".join(f"- {task}" for task in guidance["typical_work"])
        )

        with st.expander("Sources for this job-fit explanation"):
            source_links(guidance["role_sources"])
            st.caption(
                "O*NET describes U.S. occupational tasks. It is used here as a "
                "task reference, not as a Philippine licensing rule, a live vacancy, "
                "or one employer's job description."
            )

        st.subheader("What your experience level may mean")
        st.write(result["experience_guidance"])

        st.subheader("Where this job family is applied")
        st.markdown(
            "\n".join(
                f"- {context}" for context in result["application_contexts"]
            )
        )
        st.caption(
            "These are cross-industry application contexts, not claims of live "
            "vacancies in a particular employer, location, or salary band."
        )

    with ai_tab:
        st.subheader("The short answer")
        st.write(guidance["ai_explanation"])
        st.info(
            "The AI-era relevance score is a comparison tool inside this app. It is "
            "not the percentage chance that this job will survive, grow, or be hired."
        )

        st.subheader("What AI may help with")
        st.write(result["ai_opportunity"])

        st.subheader("What still needs human judgment")
        st.write(result["human_edge"])

        st.subheader("How to remain competitive")
        st.markdown(
            "\n".join(f"- {action}" for action in guidance["ai_actions"])
        )

        st.subheader("What the research says about the direction of change")
        st.write(result["future_demand"]["insight"])
        with st.expander("Sources for this AI-era explanation"):
            source_links(guidance["ai_sources"])
            st.caption(
                "Role-specific AI examples are practical inferences from the cited "
                "occupational tasks and WEF/ILO research on skill change and human–AI "
                "augmentation. They are not guarantees about a particular employer."
            )

        st.subheader("A practical proof to build")
        st.info(result["first_proof"])
        st.caption(
            "Use sanitized, public, or synthetic material. Never disclose employer, "
            "customer, employee, financial, health, security, or laboratory-sensitive data."
        )

    with skills_tab:
        st.subheader("Skills that would strengthen this match")
        st.write(
            "These are role-relevant skills where your self-rating is below the "
            "teaching target used by the synthetic model. The explanation below shows "
            "why each skill is used in the work—not only the score difference."
        )
        if result["skill_gaps"]:
            for gap_number, gap in enumerate(result["skill_gaps"], start=1):
                st.markdown(f"### {gap_number}. {gap['label']}")
                st.caption(
                    f"Your rating: {gap['current']:g}/5 · teaching target used "
                    f"for this job family: {gap['target']}/5"
                )
                st.markdown(f"**Why this skill is needed:** {gap['why']}")
                st.markdown(f"**What it can look like at work:** {gap['example']}")
                with st.expander(f"Sources for {gap['label']}"):
                    source_links(gap["sources"])
                st.divider()
        else:
            st.success(
                "Your self-ratings meet the teaching targets for this role's priority "
                "skills. The next step is to prove those ratings with a work sample "
                "and obtain feedback from a practitioner."
            )
        st.caption(
            "The reasons are tied to the cited occupational tasks. The 1–5 targets "
            "remain synthetic design assumptions—not employer requirements, Philippine "
            "occupational standards, or proof that a course is necessary. Compare the "
            "guidance with several current job descriptions before paying for training."
        )

    with learning_tab:
        st.info(
            "These are options to investigate, not mandatory requirements. A "
            "credential does not guarantee employment, promotion, salary, or skill."
        )
        for option in result["learning_options"]:
            st.subheader(option["name"])
            st.markdown(
                f"**{option['type']} · Official provider: "
                f"{option['provider']}**"
            )
            st.write(option["fit"])
            st.markdown(f"**Eligibility or access note:** {option['eligibility']}")
            st.link_button(
                f"Open official {option['provider']} page ↗",
                option["url"],
                use_container_width=True,
            )
            st.divider()

    with evidence_tab:
        current_col, future_col = st.columns(2)
        with current_col:
            st.subheader("Current-demand evidence")
            st.markdown(
                f"**{result['current_demand']['label']} · "
                f"{result['current_demand']['basis']}**"
            )
            st.write(result["current_demand"]["insight"])
            source_links(result["current_demand"]["sources"])
        with future_col:
            st.subheader("Future-demand evidence")
            st.markdown(
                f"**{result['future_demand']['label']} · "
                f"{result['future_demand']['basis']}**"
            )
            st.write(result["future_demand"]["insight"])
            source_links(result["future_demand"]["sources"])
        st.warning(
            "Demand grades are editorial summaries of cited sources—not live "
            "vacancy counts, Philippine salary forecasts, or guarantees that a "
            "specific employer is hiring."
        )


st.html(
    """
    <div id="hakbang-navigation">
      <nav class="topbar">
        <div class="brand">
          <span class="brand-mark">H</span>
          <span>Hakbang PH</span>
        </div>
        <div class="nav-pills">
          <button type="button" data-scroll-target="career-scan"
                  aria-controls="career-scan">Career scan</button>
          <button type="button" data-scroll-target="how-it-works"
                  aria-controls="how-it-works">Method</button>
          <button type="button" data-scroll-target="evidence-library"
                  aria-controls="evidence-library">Evidence</button>
          <button class="nav-cta" type="button" data-scroll-target="career-scan"
                  aria-controls="career-scan">Find my next move →</button>
        </div>
      </nav>
        <section class="hero">
          <div class="hero-copy">
            <div class="hero-kicker">Career intelligence for Filipino professionals</div>
            <h1>Your next move.<br/>Built for the AI era.</h1>
            <p>
              Match your demonstrated skills and total experience to the closest
              cross-industry job families—then investigate research-linked demand,
              AI-era work patterns, priority skill gaps, and official learning options.
            </p>
            <div class="hero-actions">
              <button class="hero-primary" type="button"
                      data-scroll-target="career-scan"
                      aria-controls="career-scan">Start my career scan →</button>
              <button class="hero-secondary" type="button"
                      data-scroll-target="how-it-works"
                      aria-controls="how-it-works">See how it works</button>
            </div>
          </div>
          <div class="hero-visual" aria-hidden="true">
            <div class="machine-glow"></div>
            <div class="orbit orbit-a"></div>
            <div class="orbit orbit-b"></div>
            <div class="orbit orbit-c"></div>
            <div class="track track-a"></div>
            <div class="track track-b"></div>
            <div class="track track-c"></div>
            <div class="machine-core">YOU</div>
            <div class="signal-card signal-one">AI opportunity ↗</div>
            <div class="signal-card signal-two">Skills → proof</div>
            <div class="signal-card signal-three">What could be next?</div>
          </div>
        </section>
        <section class="proof-strip">
          <div class="proof-quote">
            <strong>Explore. Verify. Build.</strong>
            <span>A decision aid—not a career guarantee.</span>
          </div>
          <div class="proof-chip">2,200 synthetic<br/>skill profiles</div>
          <div class="proof-chip">11 career<br/>pathways</div>
          <div class="proof-chip">12 transferable<br/>skill signals</div>
        </section>
        <div class="truth-note">
          <strong>Research prototype:</strong> industry, employer, current title,
          demographics, and protected characteristics are not model inputs. The model
          learns from synthetic skills and experience only; research claims come from
          a fixed source registry. Do not use results for hiring, promotion,
          redundancy, compensation, or another high-impact decision.
        </div>
    </div>
    <script>
      (() => {
        const navigation = document.getElementById("hakbang-navigation");
        if (!navigation) return;
        const reducedMotion = window.matchMedia(
          "(prefers-reduced-motion: reduce)"
        ).matches;
        const pause = (milliseconds) =>
          new Promise((resolve) => window.setTimeout(resolve, milliseconds));
        const revealAndScroll = async (targetId) => {
          const scroller = document.querySelector('[data-testid="stMain"]');
          let target = document.getElementById(targetId);

          if (!target && scroller) {
            const maximum = Math.max(
              0,
              scroller.scrollHeight - scroller.clientHeight
            );
            const hint = {
              "career-scan": 0,
              "how-it-works": maximum * 0.58,
              "evidence-library": maximum
            }[targetId] ?? 0;

            scroller.scrollTo({top: hint, behavior: "auto"});
            await pause(100);
            target = document.getElementById(targetId);

            if (!target) {
              const step = Math.max(500, scroller.clientHeight * 0.8);
              for (let position = 0; position <= maximum; position += step) {
                scroller.scrollTo({top: position, behavior: "auto"});
                await pause(70);
                target = document.getElementById(targetId);
                if (target) break;
              }
            }
          }

          if (target) {
            target.scrollIntoView({
              behavior: reducedMotion ? "auto" : "smooth",
              block: "start"
            });
          }
        };
        navigation.querySelectorAll("[data-scroll-target]").forEach((button) => {
          button.addEventListener("click", () => {
            revealAndScroll(button.dataset.scrollTarget);
          });
        });
      })();
    </script>
    """,
    unsafe_allow_javascript=True,
)

input_tab = st.container()
method_tab = st.container()
evidence_tab = st.container()

engine = load_engine(ENGINE_CACHE_KEY)

with input_tab:
    st.markdown('<span id="career-scan"></span>', unsafe_allow_html=True)
    st.header("Map the skills you can demonstrate today")
    st.write(
        "Use the 0–5 skill scale honestly: 0 means no experience with the skill; "
        "1 means new to the skill; 5 means you can independently deliver strong "
        "work and explain your decisions. Industry and current job title are not used."
    )
    with st.form("career_profile"):
        experience_col, explanation_col = st.columns([1, 2.5])
        with experience_col:
            years_experience = st.number_input(
                "Total years of work experience",
                min_value=0.0,
                max_value=50.0,
                value=0.0,
                step=0.5,
                help=(
                    "Used only to suggest an appropriate investigation level. "
                    "It does not prove seniority or readiness."
                ),
            )
        with explanation_col:
            st.info(
                "Your match is driven by skill alignment first. Experience, "
                "synthetic model fit, and dated demand evidence have smaller weights. "
                "No industry advantage or penalty is applied."
            )

        st.subheader("Self-assess your transferable skills")
        st.info(
            "**Zero is a valid, completed response.** Leave any skill at 0 when "
            "you have no experience with it. You do not need to move every slider. "
            "The app needs at least one honestly rated skill above 0 to compare "
            "your profile with its job catalog."
        )
        skill_columns = st.columns(3)
        skill_values: dict[str, int] = {}
        for index, (skill_key, skill) in enumerate(SKILLS.items()):
            with skill_columns[index % 3]:
                skill_values[skill_key] = st.slider(
                    skill["label"],
                    min_value=0,
                    max_value=5,
                    value=0,
                    help=(
                        "0 = no experience; 1 = new to the skill; "
                        f"5 = independently proficient. Examples: {skill['hint']}."
                    ),
                    key=f"skill_{skill_key}",
                )

        submitted = st.form_submit_button(
            "Show my supported career moves",
            type="primary",
            use_container_width=True,
        )

    st.caption(
        "Privacy: this demo does not ask for your name, age, employer, résumé, "
        "industry, current title, or protected characteristics. Inputs remain only "
        "in the current app session. "
        f"Build: {APP_BUILD_ID}."
    )

    if submitted:
        profile = {
            "years_experience": years_experience,
            **skill_values,
        }
        try:
            st.session_state["recommendations"] = engine.recommend(profile)
            st.session_state["profile"] = profile
        except ValueError as error:
            st.session_state.pop("recommendations", None)
            st.session_state.pop("profile", None)
            st.error(str(error))

    if "recommendations" in st.session_state:
        recommendations = st.session_state["recommendations"]
        st.divider()
        if len(recommendations) == 3:
            st.header("Your three closest skills-based job matches")
        elif len(recommendations) == 1:
            st.header("Your closest supported job match")
        else:
            st.header(f"Your {len(recommendations)} closest supported job matches")
        st.write(
            "Treat these as job families to investigate through current postings, "
            "practitioner conversations, work samples, and official training pages."
        )
        if len(recommendations) < 3:
            st.warning(
                "The app found fewer than three sufficiently supported pathways. "
                "It omitted lower-fit results instead of filling the list with "
                "unrelated careers."
            )
        saved_profile = st.session_state.get("profile", {})
        if saved_profile:
            st.caption(
                f"Profile signal used: {saved_profile['years_experience']:g} years "
                f"of experience · {sum(value > 0 for key, value in saved_profile.items() if key in SKILLS)} "
                "skills rated above zero · no industry or current-title input"
            )
        recommendation_tabs = st.tabs(
            [
                f"{rank}. {item['display_career']}"
                for rank, item in enumerate(
                    recommendations,
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
    st.divider()
    st.markdown('<span id="how-it-works"></span>', unsafe_allow_html=True)
    st.header("A transparent skills-first recommender")
    st.markdown(
        f"""
        1. **Skills-only synthetic model:** a {MODEL_NAME.lower()} classifier learns
           patterns from {len(engine.synthetic_profiles):,} seeded synthetic
           profiles—{SYNTHETIC_PROFILES_PER_CAREER} per job family. Its only
           features are total experience and the 12 self-rated skills.
        2. **Transparent reranking:** direct skill alignment and coverage of each
           job family's core skills carry most of the score. Experience proximity,
           synthetic model fit, and dated demand grades have smaller weights.
        3. **Fixed evidence lookup:** research, AI-era insights, official
           certifications, and courses are retrieved from a reviewed registry.
           They are not generated by the model.

        The reranking weights are:

        - direct skill alignment: **50%**
        - career-specific core-skill coverage: **15%**
        - synthetic model fit: **15%**
        - experience proximity: **8%**
        - current-demand evidence grade: **5%**
        - future-demand evidence grade: **7%**

        A result must also meet minimum skill-alignment and core-skill-coverage
        thresholds. The app can return fewer than three matches or abstain instead
        of filling the list with unrelated jobs.
        """
    )
    st.info(
        f"In the revised seeded five-fold benchmark, {MODEL_NAME.lower()} "
        f"performed best among four tested classifiers, with mean macro-F1 "
        f"{SYNTHETIC_CV_MACRO_F1:.3f} and mean accuracy "
        f"{SYNTHETIC_CV_ACCURACY:.3f}. These measure recovery of synthetic "
        "labeling rules—not real-world career accuracy."
    )
    st.subheader("Download the teaching dataset")
    csv_buffer = StringIO()
    engine.synthetic_profiles.to_csv(csv_buffer, index=False)
    st.download_button(
        f"Download {len(engine.synthetic_profiles):,} synthetic profiles (.csv)",
        data=csv_buffer.getvalue(),
        file_name="hakbang_ph_2200_skills_first_profiles.csv",
        mime="text/csv",
    )
    st.caption(
        f"Dataset/model version: {DATASET_VERSION}. No real people, résumés, "
        "employers, vacancies, or employment outcomes are included."
    )
    with st.expander("Important limitations"):
        st.markdown(
            """
            - More synthetic rows reduce sampling noise but do not add real-world truth.
            - Self-assessed skills can be inconsistent.
            - Demand grades are ordinal editorial mappings, not live job counts.
            - Global evidence does not automatically describe every Philippine region.
            - The catalogue contains only eleven job families and omits many valid careers.
            - The model has not been validated with real applicants or employment outcomes.
            - Years of experience do not prove domain depth, leadership, or seniority.
            - Certification requirements, prices, and exam versions can change.
            """
        )

with evidence_tab:
    st.divider()
    st.markdown('<span id="evidence-library"></span>', unsafe_allow_html=True)
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
