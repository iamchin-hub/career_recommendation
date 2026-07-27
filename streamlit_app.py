"""Public Streamlit interface for the Hakbang PH career recommender."""

from __future__ import annotations

from io import StringIO

import streamlit as st

from career_engine import (
    CAREER_GOALS,
    CURRENT_ROLES,
    DATASET_VERSION,
    EVIDENCE_CHECKED,
    INDUSTRIES,
    MODEL_NAME,
    SKILLS,
    SOURCES,
    SYNTHETIC_CV_ACCURACY,
    SYNTHETIC_CV_MACRO_F1,
    SYNTHETIC_PROFILES_PER_CAREER,
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
      .stApp input::placeholder,
      .stApp textarea::placeholder {
        color: #596579 !important;
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
      .nav-pills a {
        color: #d8e3f4 !important;
        text-decoration: none;
        padding: .48rem .78rem;
        border-radius: 9px;
        font-size: .83rem;
      }
      .nav-pills a:hover {background: rgba(255,255,255,.07);}
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
      .hero-actions a {
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
      div[role="tablist"] {
        gap: .3rem;
        background: rgba(11,18,32,.72);
        border: 1px solid var(--line);
        border-radius: 13px;
        padding: .3rem;
      }
      [data-testid="stTab"] {
        border-radius: 9px;
        color: #ffffff !important;
        background-color: transparent !important;
        transition: background-color .16s ease, color .16s ease;
      }
      .stApp [data-testid="stTab"] p,
      .stApp [data-testid="stTab"] span {
        color: inherit !important;
        opacity: 1 !important;
      }
      .stApp [data-testid="stTab"][aria-selected="true"] {
        color: #050914 !important;
        background-color: #ffffff !important;
      }
      .stApp [data-testid="stTab"][aria-selected="true"] p,
      .stApp [data-testid="stTab"][aria-selected="true"] span {
        color: #050914 !important;
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
      [data-testid="stExpander"] summary:hover,
      [data-testid="stExpander"] details[open] > summary:hover {
        color: #ffffff !important;
        background: #246bfe !important;
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


@st.cache_resource(show_spinner="Training the 1,000-profile synthetic model…")
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

    score_col, model_col, skill_col, role_col = st.columns(4)
    score_col.metric("Comparative score", f"{result['recommendation_score']:.1f}/100")
    model_col.metric("Synthetic model fit", f"{result['synthetic_model_fit']:.1f}%")
    skill_col.metric("Skill fit index", f"{result['skill_fit']:.1f}%")
    role_col.metric("Current-role relevance", f"{result['role_fit']:.1f}%")
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
    <nav class="topbar">
      <div class="brand">
        <span class="brand-mark">H</span>
        <span>Hakbang PH</span>
      </div>
      <div class="nav-pills">
        <a href="#career-scan">Career scan</a>
        <a href="#how-it-works">Method</a>
        <a href="#evidence-library">Evidence</a>
        <a class="nav-cta" href="#career-scan">Find my next move →</a>
      </div>
    </nav>
    <section class="hero">
      <div class="hero-copy">
        <div class="hero-kicker">Career intelligence for Filipino professionals</div>
        <h1>Your next move.<br/>Built for the AI era.</h1>
        <p>
          Turn your current job, industry experience, and transferable skills
          into three evidence-linked career hypotheses—with an AI opportunity,
          a practical portfolio proof, and an official credential to investigate.
        </p>
        <div class="hero-actions">
          <a class="hero-primary" href="#career-scan">Start my career scan →</a>
          <a class="hero-secondary" href="#how-it-works">See how it works</a>
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
      <div class="proof-chip">1,000 synthetic<br/>learning profiles</div>
      <div class="proof-chip">10 career<br/>pathways</div>
      <div class="proof-chip">Research + official<br/>credential links</div>
    </section>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="truth-note">
      <strong>Research prototype:</strong> the model now learns from 1,000
      diverse—but still synthetic—profiles. More simulated rows make the
      patterns less repetitive; they do not create real-world evidence or prove
      career outcomes. Do not use these results for hiring, promotion,
      redundancy, compensation, or another high-impact decision.
    </div>
    """,
    unsafe_allow_html=True,
)

input_tab, method_tab, evidence_tab = st.tabs(
    ["Build my profile", "How it works", "Evidence library"]
)

engine = load_engine()

with input_tab:
    st.markdown('<span id="career-scan"></span>', unsafe_allow_html=True)
    st.header("Tell us what you can demonstrate today")
    st.write(
        "Use the 1–5 skill scale honestly: 1 means new to the skill; 5 means you "
        "can independently deliver strong work and explain your decisions."
    )
    with st.form("career_profile"):
        title_col, role_col = st.columns([1.15, 1])
        with title_col:
            current_job_title = st.text_input(
                "Current job title",
                value="Customer Service Representative",
                placeholder="Example: Finance Analyst or Team Leader",
                help=(
                    "The model uses words in your job title together with the "
                    "closest role family selected beside it."
                ),
            )
        with role_col:
            current_role = st.selectbox(
                "Closest current-role family",
                options=list(CURRENT_ROLES),
                format_func=lambda role: CURRENT_ROLES[role]["label"],
                help=(
                    "Choose the family that best represents your present work, "
                    "even if your exact title is different."
                ),
            )

        context_col, experience_col, leadership_col, goal_col = st.columns(4)
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
        with leadership_col:
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
            "current_role": current_role,
            "current_job_title": current_job_title.strip(),
            "years_experience": years_experience,
            "leadership_years": leadership_years,
            "goal": goal,
            **skill_values,
        }
        try:
            st.session_state["recommendations"] = engine.recommend(profile)
            st.session_state["profile"] = profile
        except ValueError as error:
            st.error(str(error))

    if "recommendations" in st.session_state:
        st.divider()
        st.header("Your three next-move hypotheses")
        st.write(
            "Treat these as options to investigate through conversations, work "
            "samples, and current job postings—not as instructions."
        )
        saved_profile = st.session_state.get("profile", {})
        if saved_profile:
            st.caption(
                "Profile signal used: "
                f"{saved_profile['current_job_title']} · "
                f"{CURRENT_ROLES[saved_profile['current_role']]['label']} · "
                f"{INDUSTRIES[saved_profile['current_industry']]}"
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
    st.markdown('<span id="how-it-works"></span>', unsafe_allow_html=True)
    st.header("A transparent two-stage recommender")
    st.markdown(
        f"""
        1. **Synthetic model fit:** a {MODEL_NAME.lower()} classifier learns
           patterns from {len(engine.synthetic_profiles):,} seeded synthetic
           profiles—{SYNTHETIC_PROFILES_PER_CAREER} per career pathway. Current
           job title and current-role family are included alongside industry,
           experience, and skills.
        2. **Evidence reranking:** the app combines model fit with direct skill
           fit, current-role relevance, experience, industry adjacency, and
           dated demand grades.

        The reranking weights are:

        - synthetic model fit: **42%**
        - direct skill fit: **20%**
        - current-role relevance: **14%**
        - experience fit: **5%**
        - current-demand evidence grade: **5%**
        - future-demand evidence grade: **7%**
        - industry adjacency: **4%**
        - goal adjustment: **up to 3%**
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
        "Download 1,000 synthetic profiles (.csv)",
        data=csv_buffer.getvalue(),
        file_name="hakbang_ph_1000_synthetic_profiles.csv",
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
            - Job-title wording and role-family selection can be ambiguous.
            - Demand grades are ordinal editorial mappings, not live job counts.
            - Global evidence does not automatically describe every Philippine region.
            - The catalogue contains only ten pathways and omits many valid careers.
            - Practitioner accounts may contain selection and survivorship bias.
            - Certification requirements, prices, and exam versions can change.
            """
        )

with evidence_tab:
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
