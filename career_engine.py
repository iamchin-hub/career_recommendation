"""Explainable career recommendation engine for the Hakbang PH Streamlit app.

The model is trained only on 1,000 deterministic synthetic profiles. Its output
is a comparative exploration aid, not a hiring probability or employment forecast.
All research and credential text comes from the fixed, source-linked registry
below; no generative AI is used at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 20260725
SYNTHETIC_PROFILES_PER_CAREER = 100
DATASET_VERSION = "PH-STREAMLIT-SYN-1000-2026.07.28"
EVIDENCE_CHECKED = "27 July 2026"
MODEL_NAME = "Random Forest"
SYNTHETIC_CV_MACRO_F1 = 0.8263
SYNTHETIC_CV_ACCURACY = 0.8270

INDUSTRIES = {
    "it_bpo": "IT–BPM / BPO",
    "financial_services": "Banking & financial services",
    "manufacturing_logistics": "Manufacturing & logistics",
    "retail_ecommerce": "Retail & e-commerce",
    "government_public": "Government & public service",
    "healthcare": "Healthcare",
    "education": "Education",
    "tourism_hospitality": "Tourism & hospitality",
    "professional_services": "Professional services",
    "other": "Other industry",
}

CURRENT_ROLES = {
    "customer_service": {
        "label": "Customer service / contact center",
        "titles": [
            "Customer Service Representative",
            "Contact Center Agent",
            "Customer Support Specialist",
            "Technical Support Representative",
            "Client Services Associate",
        ],
    },
    "team_lead_supervisor": {
        "label": "Team leader / supervisor",
        "titles": [
            "Team Leader",
            "Operations Supervisor",
            "Customer Service Supervisor",
            "Shift Supervisor",
            "Unit Supervisor",
        ],
    },
    "admin_operations": {
        "label": "Administration / operations coordination",
        "titles": [
            "Administrative Officer",
            "Operations Coordinator",
            "Office Administrator",
            "Business Support Specialist",
            "Operations Associate",
        ],
    },
    "data_reporting": {
        "label": "Data, reporting, or business analysis",
        "titles": [
            "Data Analyst",
            "Reporting Analyst",
            "Business Analyst",
            "Management Information Analyst",
            "Business Intelligence Analyst",
        ],
    },
    "software_it_support": {
        "label": "Software development / IT support",
        "titles": [
            "Software Developer",
            "Application Support Analyst",
            "IT Support Specialist",
            "Systems Support Engineer",
            "Technical Support Engineer",
        ],
    },
    "network_systems": {
        "label": "Network, systems, or cloud administration",
        "titles": [
            "Network Administrator",
            "Systems Administrator",
            "Cloud Support Engineer",
            "Infrastructure Engineer",
            "Platform Operations Analyst",
        ],
    },
    "cyber_compliance": {
        "label": "Cybersecurity, risk, or compliance",
        "titles": [
            "Security Operations Analyst",
            "Information Security Analyst",
            "Risk and Compliance Analyst",
            "IT Auditor",
            "Governance Risk and Compliance Associate",
        ],
    },
    "project_coordination": {
        "label": "Project or program coordination",
        "titles": [
            "Project Coordinator",
            "Project Officer",
            "Program Coordinator",
            "PMO Analyst",
            "Scrum Master",
        ],
    },
    "marketing_content": {
        "label": "Marketing, content, or communications",
        "titles": [
            "Marketing Specialist",
            "Digital Marketing Associate",
            "Content Strategist",
            "Social Media Specialist",
            "Communications Officer",
        ],
    },
    "sales_business": {
        "label": "Sales / business development",
        "titles": [
            "Account Executive",
            "Sales Specialist",
            "Business Development Officer",
            "Relationship Manager",
            "Key Account Associate",
        ],
    },
    "hr_recruitment": {
        "label": "Human resources / recruitment",
        "titles": [
            "HR Generalist",
            "Recruitment Specialist",
            "Talent Acquisition Associate",
            "HR Business Partner",
            "Learning and Development Specialist",
        ],
    },
    "accounting": {
        "label": "Accounting / bookkeeping",
        "titles": [
            "Accountant",
            "Bookkeeper",
            "Accounts Payable Analyst",
            "General Ledger Accountant",
            "Audit Associate",
        ],
    },
    "finance_planning": {
        "label": "Finance, budgeting, or planning",
        "titles": [
            "Financial Analyst",
            "Budget Analyst",
            "Planning Analyst",
            "Commercial Finance Analyst",
            "Management Accountant",
        ],
    },
    "supply_logistics": {
        "label": "Supply chain / logistics",
        "titles": [
            "Supply Chain Coordinator",
            "Logistics Analyst",
            "Inventory Planner",
            "Demand Planner",
            "Warehouse Operations Analyst",
        ],
    },
    "quality_process": {
        "label": "Quality / process improvement",
        "titles": [
            "Quality Analyst",
            "Process Improvement Specialist",
            "Continuous Improvement Analyst",
            "Quality Assurance Specialist",
            "Business Process Analyst",
        ],
    },
    "product_ux_design": {
        "label": "Product, UX, or design",
        "titles": [
            "UX Researcher",
            "UX Designer",
            "Product Designer",
            "Product Analyst",
            "Service Designer",
        ],
    },
    "education_training": {
        "label": "Education / professional training",
        "titles": [
            "Teacher",
            "Corporate Trainer",
            "Instructional Designer",
            "Learning Facilitator",
            "Training Specialist",
        ],
    },
    "healthcare_services": {
        "label": "Healthcare / health administration",
        "titles": [
            "Healthcare Administrator",
            "Medical Services Coordinator",
            "Clinical Data Associate",
            "Healthcare Support Specialist",
            "Patient Services Officer",
        ],
    },
    "hospitality_tourism": {
        "label": "Hospitality / tourism",
        "titles": [
            "Hotel Operations Associate",
            "Guest Services Officer",
            "Travel Consultant",
            "Tourism Officer",
            "Restaurant Supervisor",
        ],
    },
    "other": {
        "label": "Another role family",
        "titles": [
            "Business Associate",
            "Professional Services Associate",
            "Program Assistant",
            "Specialist",
            "Consultant",
        ],
    },
}

SKILLS = {
    "data_analytics": {
        "label": "Data & analytics",
        "hint": "Spreadsheets, reporting, SQL, statistics, and dashboards",
    },
    "technology": {
        "label": "Technology",
        "hint": "Software, systems, cloud, coding, and technical troubleshooting",
    },
    "communication": {
        "label": "Communication",
        "hint": "Writing, presenting, negotiation, and stakeholder alignment",
    },
    "project_management": {
        "label": "Project delivery",
        "hint": "Planning, risk, scope, coordination, and agile ways of working",
    },
    "creative_design": {
        "label": "Creative & design",
        "hint": "Design, campaigns, prototyping, content, and ideation",
    },
    "finance": {
        "label": "Finance",
        "hint": "Budgeting, accounting, forecasting, and commercial analysis",
    },
    "people_hr": {
        "label": "People & HR",
        "hint": "Coaching, talent, workforce planning, and employee experience",
    },
    "operations": {
        "label": "Operations",
        "hint": "Process improvement, logistics, quality, and service delivery",
    },
    "customer_experience": {
        "label": "Customer experience",
        "hint": "Research, service design, customer success, and voice of customer",
    },
}

NUMERIC_FEATURES = [
    "years_experience",
    "leadership_years",
    *SKILLS.keys(),
]
TEXT_FEATURE = "current_job_title"
CATEGORICAL_FEATURES = ["current_industry", "current_role"]
CAREER_GOALS = {
    "future_ready": "Build skills for an AI-shaped future",
    "leadership": "Move toward leadership",
    "sector_switch": "Switch industry or sector",
    "build_specialty": "Deepen a specialist skill",
}

SOURCES = {
    "psa_lfs": {
        "name": "May 2026 Labor Force Survey",
        "owner": "Philippine Statistics Authority",
        "url": "https://psa.gov.ph/statistics/labor-force-survey?vcode=sl76S8",
        "published": "8 July 2026",
    },
    "dole_forecast": {
        "name": "Jobs and Labor Market Forecast",
        "owner": "DOLE Bureau of Local Employment",
        "url": "https://ble.dole.gov.ph/jobs-and-labor-market-forecast/",
        "published": "2023–2025 release",
    },
    "tesda_5ir": {
        "name": "TVET Skills Insights: 5th Industrial Revolution",
        "owner": "TESDA",
        "url": "https://www.tesda.gov.ph/Uploads/File/SkillInsights/2025/TVET%20Skills%20Insights%20Report%20_%205th%20Industrial%20Revolution.pdf",
        "published": "2025",
    },
    "wef_jobs": {
        "name": "Future of Jobs 2025 — Jobs Outlook",
        "owner": "World Economic Forum",
        "url": "https://www.weforum.org/publications/the-future-of-jobs-report-2025/in-full/2-jobs-outlook/",
        "published": "8 January 2025",
    },
    "wef_skills": {
        "name": "Future of Jobs 2025 — Skills Outlook",
        "owner": "World Economic Forum",
        "url": "https://www.weforum.org/publications/the-future-of-jobs-report-2025/in-full/3-skills-outlook/",
        "published": "8 January 2025",
    },
    "wef_industry": {
        "name": "Future of Jobs 2025 — Industry Insights",
        "owner": "World Economic Forum",
        "url": "https://www.weforum.org/publications/the-future-of-jobs-report-2025/in-full/5-region-economy-and-industry-insights/",
        "published": "8 January 2025",
    },
    "ilo_genai": {
        "name": "Generative AI and Jobs: A Refined Global Index",
        "owner": "International Labour Organization",
        "url": "https://www.ilo.org/publications/generative-ai-and-jobs-refined-global-index-occupational-exposure",
        "published": "20 May 2025",
    },
    "pmi_talent": {
        "name": "PMP salary and talent survey",
        "owner": "Project Management Institute",
        "url": "https://www.pmi.org/about/press-media/2025/pmp-certification-holders-build-career-momentum-and-experience-earning-advantage-pmi-survey-finds",
        "published": "13 November 2025",
    },
    "isc2_cc": {
        "name": "Who earns the ISC2 CC?",
        "owner": "ISC2",
        "url": "https://www.isc2.org/insights/2025/11/who-earns-the-isc2-certified-in-cybersecurity-certification",
        "published": "3 November 2025",
    },
    "shrm_people": {
        "name": "People Analytics Specialty Credential",
        "owner": "Society for Human Resource Management",
        "url": "https://www.shrm.org/credentials/specialty-credentials/people-analytics-credential",
        "published": f"Checked {EVIDENCE_CHECKED}",
    },
    "ascm_cpim": {
        "name": "Certified in Planning and Inventory Management",
        "owner": "Association for Supply Chain Management",
        "url": "https://www.ascm.org/learning-development/certifications-credentials/cpim/",
        "published": f"Checked {EVIDENCE_CHECKED}",
    },
}


def demand(
    label: str,
    score: float,
    basis: str,
    insight: str,
    sources: list[str],
) -> dict[str, Any]:
    return {
        "label": label,
        "score": score,
        "basis": basis,
        "insight": insight,
        "sources": sources,
    }


PORTFOLIO_CAVEAT = (
    "A credential can validate knowledge; it does not replace role-relevant "
    "projects, supervised practice, or measurable work outcomes."
)

CAREERS: dict[str, dict[str, Any]] = {
    "data_bi_analyst": {
        "title": "Data & Business Intelligence Analyst",
        "summary": (
            "Turn operational and commercial data into dashboards, decisions, "
            "and measurable business improvements."
        ),
        "min_experience": 2,
        "preferred_industries": [
            "financial_services",
            "it_bpo",
            "retail_ecommerce",
        ],
        "current_demand": demand(
            "Strong",
            0.8,
            "Sector-to-role inference",
            "DOLE identifies IT–BPM/BPO as an in-demand Philippine sector, while "
            "TESDA's 2025 skills report names big-data work among high-growth "
            "occupations. The BI mapping is an inference, not a vacancy count.",
            ["dole_forecast", "tesda_5ir"],
        ),
        "future_demand": demand(
            "Very strong",
            1.0,
            "Global directional evidence",
            "WEF's employer survey places Big Data Specialists first among the "
            "fastest-growing roles through 2030 and AI and big data first among "
            "rising skills.",
            ["wef_jobs", "wef_skills"],
        ),
        "ai_opportunity": (
            "Use copilots to draft measures, explain variance, and accelerate "
            "exploration, then own metric definitions, data quality, and decision context."
        ),
        "human_edge": (
            "Stakeholder framing and checking whether a statistically correct "
            "output answers the real business question."
        ),
        "first_proof": (
            "Build one decision dashboard with a documented data dictionary, "
            "AI-use log, and before/after business metric."
        ),
        "certification": {
            "name": "Microsoft Certified: Power BI Data Analyst Associate",
            "issuer": "Microsoft",
            "url": "https://learn.microsoft.com/en-us/credentials/certifications/data-analyst-associate/",
            "eligibility": "Intermediate credential; review the official PL-300 exam page.",
            "why_it_fits": (
                "PL-300 assesses data preparation, modeling, visualization, "
                "analysis, management, and security in Power BI."
            ),
            "practitioner": "Sarah Krusleski · Power BI practitioner",
            "practitioner_insight": (
                "After earning PL-300, she reported recruiter messages for roles "
                "that explicitly required it and inquiries about teaching Power "
                "BI. She also says certification is not mandatory."
            ),
            "source_type": "Public first-person account",
            "practitioner_url": (
                "https://www.linkedin.com/posts/sekrusleski_"
                "what-will-passing-the-pl-300-microsoft-power-"
                "activity-7269734663479279616-quT4"
            ),
            "caveat": PORTFOLIO_CAVEAT,
        },
    },
    "cybersecurity_analyst": {
        "title": "Cybersecurity Analyst",
        "summary": (
            "Monitor threats, investigate incidents, strengthen controls, and "
            "help organizations manage digital risk."
        ),
        "min_experience": 2,
        "preferred_industries": [
            "it_bpo",
            "financial_services",
            "government_public",
        ],
        "current_demand": demand(
            "Strong",
            0.8,
            "Direct role evidence",
            "ISC2 reports entry-level CC holders in analyst and "
            "security-operations pathways; DOLE separately identifies IT–BPM/BPO "
            "as an in-demand Philippine sector.",
            ["isc2_cc", "dole_forecast"],
        ),
        "future_demand": demand(
            "Very strong",
            1.0,
            "Global directional evidence",
            "WEF lists Information Security Analysts among the 15 fastest-growing "
            "roles and networks and cybersecurity as the second-fastest-rising "
            "skill group through 2030.",
            ["wef_jobs", "wef_skills"],
        ),
        "ai_opportunity": (
            "Apply AI to alert triage, threat research, and control documentation "
            "while testing AI systems for prompt injection, data leakage, and model abuse."
        ),
        "human_edge": (
            "Risk judgment, incident accountability, and adversarial reasoning "
            "when automated output is incomplete or misleading."
        ),
        "first_proof": (
            "Create an incident-response lab and publish a redacted write-up "
            "covering detection, evidence, containment, and lessons learned."
        ),
        "certification": {
            "name": "ISC2 Certified in Cybersecurity (CC)",
            "issuer": "ISC2",
            "url": "https://www.isc2.org/certifications/cc",
            "eligibility": (
                "No prior work experience is required. Review the official page "
                "for the current exam outline."
            ),
            "why_it_fits": (
                "CC validates foundational security principles for an entry or "
                "adjacent career move."
            ),
            "practitioner": "Lance Rosengarten, CC · SOC Analyst",
            "practitioner_insight": (
                "He wrote that, weeks after passing CC, he entered a GRC "
                "internship and was later offered a SOC Analyst Team Lead role."
            ),
            "source_type": "Issuer-published holder account",
            "practitioner_url": (
                "https://www.isc2.org/Insights/2024/02/"
                "My-Journey-into-Cybersecurity-With-ISC2"
            ),
            "caveat": (
                "This sequence does not prove that CC caused the outcome. He "
                "also used labs, self-study, and an internship."
            ),
        },
    },
    "cloud_solutions_engineer": {
        "title": "Cloud Solutions Engineer",
        "summary": (
            "Design reliable cloud environments and improve the cost, security, "
            "and performance of digital systems."
        ),
        "min_experience": 3,
        "preferred_industries": [
            "it_bpo",
            "financial_services",
            "professional_services",
        ],
        "current_demand": demand(
            "Strong",
            0.8,
            "Sector-to-role inference",
            "DOLE identifies IT–BPM/BPO as in demand. Cloud engineering is "
            "treated as enabling infrastructure, not counted as a live national "
            "vacancy total.",
            ["dole_forecast", "tesda_5ir"],
        ),
        "future_demand": demand(
            "Very strong",
            1.0,
            "Sector-to-role inference",
            "WEF says information and technology services employers expect "
            "near-universal AI and information-processing adoption by 2030. "
            "Cloud-architecture demand is an inference from that adoption.",
            ["wef_industry"],
        ),
        "ai_opportunity": (
            "Design the governed data, identity, security, observability, and "
            "cost controls that let teams run AI workloads reliably."
        ),
        "human_edge": (
            "Architectural trade-offs across resilience, security, latency, "
            "cost, and regulation."
        ),
        "first_proof": (
            "Deploy a retrieval-based AI service with least-privilege access, "
            "monitoring, a cost budget, and an architecture decision record."
        ),
        "certification": {
            "name": "AWS Certified Solutions Architect – Associate",
            "issuer": "Amazon Web Services",
            "url": (
                "https://aws.amazon.com/certification/"
                "certified-solutions-architect-associate/"
            ),
            "eligibility": (
                "AWS recommends prior hands-on experience; review the official exam guide."
            ),
            "why_it_fits": (
                "The credential validates design of secure, resilient, "
                "high-performing, and cost-optimized AWS solutions."
            ),
            "practitioner": "Siddharth Pasumarthy · AWS Solutions Architect",
            "practitioner_insight": (
                "He says hands-on labs broadened his technical range; after more "
                "than a year of experience and the certification, he accepted "
                "an AWS Solutions Architect offer."
            ),
            "source_type": "Issuer-published holder account",
            "practitioner_url": (
                "https://aws.amazon.com/blogs/training-and-certification/"
                "steps-to-start-your-aws-certification-journey/"
            ),
            "caveat": (
                "The holder had substantial self-learning and hands-on platform "
                "experience; the credential was one part of the transition."
            ),
        },
    },
    "project_manager": {
        "title": "Project Manager",
        "summary": (
            "Lead cross-functional work, align stakeholders, manage risk, and "
            "deliver business outcomes across industries."
        ),
        "min_experience": 4,
        "preferred_industries": [
            "professional_services",
            "it_bpo",
            "manufacturing_logistics",
        ],
        "current_demand": demand(
            "Strong",
            0.8,
            "Global directional evidence",
            "PMI's 2025 research describes project talent as cross-industry and "
            "reports sustained global demand. This is not a Philippines-only "
            "vacancy measure.",
            ["pmi_talent"],
        ),
        "future_demand": demand(
            "Strong",
            0.8,
            "Global directional evidence",
            "PMI projects that the world may need up to 30 million additional "
            "project professionals by 2035 as organizations deliver AI and other "
            "transformations.",
            ["pmi_talent"],
        ),
        "ai_opportunity": (
            "Use AI for draft plans, status synthesis, risk prompts, and meeting "
            "follow-through while keeping humans accountable for scope, value, and escalation."
        ),
        "human_edge": (
            "Negotiation, change leadership, judgment under uncertainty, and "
            "ownership of outcomes."
        ),
        "first_proof": (
            "Lead a small AI-enabled process change with a benefits baseline, "
            "risk register, adoption plan, and post-implementation review."
        ),
        "certification": {
            "name": "Project Management Professional (PMP)®",
            "issuer": "Project Management Institute",
            "url": "https://www.pmi.org/certifications/project-management-pmp",
            "eligibility": (
                "Experience and training requirements apply; consider CAPM if "
                "you are not yet eligible."
            ),
            "why_it_fits": (
                "PMP tests people, process, and business-environment capabilities "
                "across predictive, agile, and hybrid delivery."
            ),
            "practitioner": "14,628 project professionals · PMI salary survey",
            "practitioner_insight": (
                "PMI's 2025 survey across 21 countries reported a 17% higher "
                "median salary for PMP holders than non-certified respondents."
            ),
            "source_type": "Issuer survey",
            "practitioner_url": SOURCES["pmi_talent"]["url"],
            "caveat": (
                "This is an association, not proof that PMP caused the pay "
                "difference. Experience, role, geography, and employer may contribute."
            ),
        },
    },
    "digital_marketing_strategist": {
        "title": "Digital Marketing & E-commerce Strategist",
        "summary": (
            "Combine audience insight, content, paid media, and performance data "
            "to grow digital revenue."
        ),
        "min_experience": 2,
        "preferred_industries": [
            "retail_ecommerce",
            "professional_services",
            "tourism_hospitality",
        ],
        "current_demand": demand(
            "Strong",
            0.8,
            "Sector-to-role inference",
            "DOLE identifies services and tourism among in-demand Philippine "
            "sectors. Digital-marketing demand is inferred from those sectors "
            "and e-commerce activity, not a live count of vacancies.",
            ["dole_forecast"],
        ),
        "future_demand": demand(
            "Strong",
            0.8,
            "Global directional evidence",
            "WEF expects marketing and media skills to grow in importance "
            "through technological change, while generative AI also increases "
            "task automation and role redesign.",
            ["wef_skills", "ilo_genai"],
        ),
        "ai_opportunity": (
            "Use AI for creative variants, audience research, and campaign "
            "analysis, then differentiate through positioning, experimentation, "
            "consent, and measurement."
        ),
        "human_edge": (
            "Brand judgment, cultural context, customer empathy, and knowing when "
            "apparent performance is a measurement artifact."
        ),
        "first_proof": (
            "Run a small campaign experiment with human-reviewed AI variants, a "
            "pre-registered hypothesis, and a clean results readout."
        ),
        "certification": {
            "name": "Meta Certified Digital Marketing Associate",
            "issuer": "Meta Blueprint",
            "url": (
                "https://www.facebookblueprint.com/student/path/"
                "517001-get-certified-as-digital-marketing-associate"
            ),
            "eligibility": (
                "Entry-level certification; confirm current language and exam availability."
            ),
            "why_it_fits": (
                "The credential covers Meta advertising fundamentals, targeting, "
                "creative, optimization, and measurement."
            ),
            "practitioner": "Ahmad · Meta-certified learner",
            "practitioner_insight": (
                "His public first-person account says passing the certificate "
                "alone produced no offers; applying the learning in visible work "
                "mattered more."
            ),
            "source_type": "Public first-person account",
            "practitioner_url": (
                "https://medium.com/write-a-catalyst/"
                "i-passed-the-meta-certification-and-waited-for-my-life-to-change-"
                "it-didnt-until-i-did-this-1992e64e7404"
            ),
            "caveat": (
                "The author's identity and outcome are not independently "
                "verified. The account is included as a counterexample to "
                "credential guarantees."
            ),
        },
    },
    "people_analytics_specialist": {
        "title": "People Analytics Specialist",
        "summary": (
            "Use workforce data to improve retention, performance, and strategic "
            "talent decisions."
        ),
        "min_experience": 3,
        "preferred_industries": [
            "it_bpo",
            "professional_services",
            "financial_services",
        ],
        "current_demand": demand(
            "Moderate",
            0.6,
            "Sector-to-role inference",
            "SHRM documents an active professional pathway for people analytics, "
            "but the reviewed sources do not provide a Philippine job count for "
            "this specialist title.",
            ["shrm_people"],
        ),
        "future_demand": demand(
            "Strong",
            0.8,
            "Global directional evidence",
            "WEF places talent management among the ten fastest-rising skills "
            "through 2030; combining it with data literacy supports a "
            "forward-looking HR specialization.",
            ["wef_skills"],
        ),
        "ai_opportunity": (
            "Use AI to surface workforce patterns and skills gaps, then test for "
            "bias, privacy risk, weak proxies, and unsupported causal conclusions."
        ),
        "human_edge": (
            "Employee trust, ethical interpretation, organizational context, and "
            "responsible decisions about people."
        ),
        "first_proof": (
            "Create an anonymized retention or skills dashboard with a privacy "
            "note, bias checks, and clearly separated correlation versus causation."
        ),
        "certification": {
            "name": "SHRM People Analytics Specialty Credential",
            "issuer": "Society for Human Resource Management",
            "url": SOURCES["shrm_people"]["url"],
            "eligibility": (
                "A structured program and final knowledge assessment are required."
            ),
            "why_it_fits": (
                "SHRM's program connects people-data literacy, metrics, analysis, "
                "and action to HR decisions."
            ),
            "practitioner": "Cathy Evans, SHRM-SCP · HR professional",
            "practitioner_insight": (
                "Her issuer-published testimonial describes moving from feeling "
                "intimidated by numerical storytelling to feeling more confident "
                "and motivated to go deeper."
            ),
            "source_type": "Issuer-published holder account",
            "practitioner_url": (
                "https://www.shrm.org/gl/shop/product.html/"
                "shrm-people-analytics-specialty-credential-p"
            ),
            "caveat": (
                "This is issuer-selected learning feedback, not an independently "
                "measured employment outcome."
            ),
        },
    },
    "supply_chain_analyst": {
        "title": "Supply Chain Analyst",
        "summary": (
            "Improve inventory, planning, sourcing, and logistics decisions with "
            "process knowledge and data."
        ),
        "min_experience": 2,
        "preferred_industries": [
            "manufacturing_logistics",
            "retail_ecommerce",
            "other",
        ],
        "current_demand": demand(
            "Strong",
            0.8,
            "Sector-to-role inference",
            "DOLE's national forecast covers employment-generating sectors and "
            "TESDA describes continuing digital transformation in operations. "
            "The analyst-title score is a structured inference.",
            ["dole_forecast", "tesda_5ir"],
        ),
        "future_demand": demand(
            "Strong",
            0.8,
            "Global directional evidence",
            "Digitalization, risk, and planning complexity support continued "
            "demand for analytical supply-chain skills; WEF also expects resource "
            "management and operations skills to rise.",
            ["wef_skills", "ascm_cpim"],
        ),
        "ai_opportunity": (
            "Use AI for demand scenarios, exception detection, and inventory "
            "recommendations while retaining judgment about disruption, service "
            "levels, and supplier constraints."
        ),
        "human_edge": (
            "Cross-functional trade-offs, exception handling, supplier "
            "relationships, and operational accountability."
        ),
        "first_proof": (
            "Build a forecast-versus-actual review with an exception queue, "
            "service-level impact, and documented human override rules."
        ),
        "certification": {
            "name": "APICS Certified in Planning and Inventory Management (CPIM)",
            "issuer": "Association for Supply Chain Management",
            "url": SOURCES["ascm_cpim"]["url"],
            "eligibility": (
                "Confirm the current exam version and learning-system options."
            ),
            "why_it_fits": (
                "CPIM covers planning, inventory, demand, supply, quality, "
                "continuous improvement, and technology."
            ),
            "practitioner": "James Tilton, CPIM · Director of Materials",
            "practitioner_insight": (
                "His issuer-published testimonial says the credential enabled "
                "him to pursue a broader path across operations, inventory, and "
                "supply chain."
            ),
            "source_type": "Issuer-published holder account",
            "practitioner_url": SOURCES["ascm_cpim"]["url"],
            "caveat": (
                "This is an issuer-selected testimonial. Treat it as a possible "
                "pathway, not an expected result."
            ),
        },
    },
    "fpa_analyst": {
        "title": "Financial Planning & Analysis Analyst",
        "summary": (
            "Translate financial and operating performance into forecasts, "
            "scenarios, and decisions for business leaders."
        ),
        "min_experience": 3,
        "preferred_industries": [
            "financial_services",
            "retail_ecommerce",
            "professional_services",
        ],
        "current_demand": demand(
            "Moderate",
            0.6,
            "Mixed evidence",
            "The Philippine services economy is large, but the PSA survey does "
            "not isolate FP&A. This signal is deliberately moderate because the "
            "reviewed national data is industry-level.",
            ["psa_lfs"],
        ),
        "future_demand": demand(
            "Mixed",
            0.6,
            "Mixed evidence",
            "WEF expects accounting and audit roles to decline while analytical "
            "thinking remains a core skill. The opportunity is a move away from "
            "transaction processing toward scenarios, controls, and business partnering.",
            ["wef_jobs", "wef_skills"],
        ),
        "ai_opportunity": (
            "Use AI to draft scenarios, investigate variance, and summarize "
            "drivers while owning assumptions, controls, and advice to decision-makers."
        ),
        "human_edge": (
            "Commercial judgment, governance, challenge of assumptions, and "
            "translating numbers into accountable choices."
        ),
        "first_proof": (
            "Create a driver-based forecast with three scenarios, sensitivity "
            "analysis, control checks, and a one-page management recommendation."
        ),
        "certification": {
            "name": "Certified Management Accountant (CMA)®",
            "issuer": "Institute of Management Accountants",
            "url": "https://www.imanet.org/ima-certifications/cma-certification",
            "eligibility": (
                "Education, experience, membership, and two exam parts are required."
            ),
            "why_it_fits": (
                "CMA covers management accounting, planning, performance, "
                "analytics, controls, and strategic finance."
            ),
            "practitioner": "Dylan Kady, CMA · Senior Financial Analyst",
            "practitioner_insight": (
                "His issuer-published account reports applying pricing and "
                "forecasting learning and later receiving broader opportunities."
            ),
            "source_type": "Issuer-published holder account",
            "practitioner_url": (
                "https://www.imanet.org/en/Newsletters/Inside-IMA/2018/April/"
                "myCMA-Dylan-Kady"
            ),
            "caveat": (
                "This is one issuer-published career story; education, experience, "
                "networking, and employer context also contributed."
            ),
        },
    },
    "customer_experience_manager": {
        "title": "Customer Experience Manager",
        "summary": (
            "Lead customer insight, journey improvement, service metrics, and "
            "cross-functional experience programs."
        ),
        "min_experience": 4,
        "preferred_industries": [
            "it_bpo",
            "retail_ecommerce",
            "tourism_hospitality",
        ],
        "current_demand": demand(
            "Strong",
            0.8,
            "Sector-to-role inference",
            "DOLE identifies IT–BPM/BPO and services as in-demand Philippine "
            "sectors. CX leadership is inferred from those sectors rather than "
            "counted as a standalone occupation.",
            ["dole_forecast"],
        ),
        "future_demand": demand(
            "Mixed",
            0.6,
            "Mixed evidence",
            "GenAI can automate service tasks, but ILO's task-based evidence "
            "indicates transformation is more common than full job replacement. "
            "CX leadership becomes more valuable when it designs the human–AI handoff.",
            ["ilo_genai"],
        ),
        "ai_opportunity": (
            "Use AI for conversation summaries, routing, and self-service while "
            "redesigning escalation, quality checks, accessibility, and recovery "
            "for high-stakes interactions."
        ),
        "human_edge": (
            "Empathy, service recovery, organizational change, and accountability "
            "for customer trust."
        ),
        "first_proof": (
            "Map one service journey, identify safe automation points, and "
            "define handoff, quality, accessibility, and failure-recovery measures."
        ),
        "certification": {
            "name": "Certified Customer Experience Professional (CCXP)",
            "issuer": "Customer Experience Professionals Association",
            "url": "https://cxpaglobal.org/get-certified",
            "eligibility": (
                "Education and multi-competency CX experience requirements apply."
            ),
            "why_it_fits": (
                "CCXP assesses strategy, insight, design, measurement, and "
                "culture across customer experience."
            ),
            "practitioner": "Mariana De Marchi, CCXP · CX leader",
            "practitioner_insight": (
                "She reports finding and securing a role through the CXPA "
                "community. Her account points to the network around the "
                "credential, not the exam alone."
            ),
            "source_type": "Issuer-published holder account",
            "practitioner_url": "https://cxpaglobal.org/",
            "caveat": (
                "The reported outcome is tied to association participation and "
                "community access; it should not be attributed solely to certification."
            ),
        },
    },
    "ux_researcher": {
        "title": "User Experience Researcher",
        "summary": (
            "Study users, synthesize evidence, and help product teams design "
            "more useful and inclusive digital services."
        ),
        "min_experience": 2,
        "preferred_industries": [
            "it_bpo",
            "retail_ecommerce",
            "professional_services",
        ],
        "current_demand": demand(
            "Moderate",
            0.6,
            "Global directional evidence",
            "The reviewed Philippine sources do not provide a current UX "
            "researcher count. The signal relies on global evidence and is "
            "therefore kept moderate.",
            ["wef_industry"],
        ),
        "future_demand": demand(
            "Mixed",
            0.6,
            "Mixed evidence",
            "WEF expects design and UX skills to rise overall, but some "
            "technology-services employers expect lower demand. Research rigor "
            "and AI-product evaluation are the more defensible specialization.",
            ["wef_skills", "wef_industry"],
        ),
        "ai_opportunity": (
            "Use AI to accelerate recruiting materials, synthesis, and "
            "prototypes while preserving research validity, consent, "
            "accessibility, and direct contact with users."
        ),
        "human_edge": (
            "Choosing the right question, noticing weak evidence, facilitating "
            "difficult conversations, and representing affected users."
        ),
        "first_proof": (
            "Run a five-user study of an AI feature, document consent and "
            "limitations, and show which product decision changed because of evidence."
        ),
        "certification": {
            "name": "NN/g UX Certification",
            "issuer": "Nielsen Norman Group",
            "url": "https://www.nngroup.com/ux-certification/",
            "eligibility": (
                "Five eligible live courses and corresponding exams are required."
            ),
            "why_it_fits": (
                "The program can combine research, design, and AI-related "
                "courses and publishes its requirements and cost."
            ),
            "practitioner": "Corey Nunez · UX-certified practitioner",
            "practitioner_insight": (
                "His issuer-published testimonial says the credential added "
                "credibility to his decisions in industry."
            ),
            "source_type": "Issuer-published holder account",
            "practitioner_url": "https://www.nngroup.com/ux-certification/",
            "caveat": (
                "NN/g states that the program is not accredited. Holder comments "
                "are selected testimonials and the listed investment is substantial."
            ),
        },
    },
}

PROTOTYPES = {
    "data_bi_analyst": {
        "industries": [
            "financial_services",
            "it_bpo",
            "retail_ecommerce",
            "professional_services",
        ],
        "current_roles": [
            "data_reporting",
            "admin_operations",
            "quality_process",
        ],
        "experience": 4.5,
        "leadership": 0.8,
        "skills": [0.92, 0.78, 0.66, 0.53, 0.34, 0.56, 0.31, 0.52, 0.48],
    },
    "cybersecurity_analyst": {
        "industries": [
            "it_bpo",
            "financial_services",
            "government_public",
            "professional_services",
        ],
        "current_roles": [
            "cyber_compliance",
            "software_it_support",
            "network_systems",
        ],
        "experience": 5.0,
        "leadership": 0.7,
        "skills": [0.60, 0.96, 0.62, 0.55, 0.20, 0.30, 0.20, 0.54, 0.37],
    },
    "cloud_solutions_engineer": {
        "industries": [
            "it_bpo",
            "financial_services",
            "professional_services",
            "retail_ecommerce",
        ],
        "current_roles": [
            "network_systems",
            "software_it_support",
            "project_coordination",
        ],
        "experience": 5.5,
        "leadership": 1.0,
        "skills": [0.66, 0.97, 0.66, 0.68, 0.25, 0.28, 0.20, 0.63, 0.42],
    },
    "project_manager": {
        "industries": [
            "professional_services",
            "it_bpo",
            "manufacturing_logistics",
            "government_public",
        ],
        "current_roles": [
            "project_coordination",
            "team_lead_supervisor",
            "admin_operations",
            "quality_process",
        ],
        "experience": 8.0,
        "leadership": 3.5,
        "skills": [0.58, 0.57, 0.92, 0.97, 0.35, 0.44, 0.55, 0.69, 0.72],
    },
    "digital_marketing_strategist": {
        "industries": [
            "retail_ecommerce",
            "it_bpo",
            "tourism_hospitality",
            "professional_services",
        ],
        "current_roles": [
            "marketing_content",
            "sales_business",
            "customer_service",
        ],
        "experience": 4.5,
        "leadership": 1.0,
        "skills": [0.72, 0.62, 0.88, 0.61, 0.96, 0.35, 0.28, 0.42, 0.82],
    },
    "people_analytics_specialist": {
        "industries": [
            "it_bpo",
            "professional_services",
            "financial_services",
            "healthcare",
        ],
        "current_roles": [
            "hr_recruitment",
            "data_reporting",
            "team_lead_supervisor",
        ],
        "experience": 5.5,
        "leadership": 1.2,
        "skills": [0.84, 0.58, 0.86, 0.58, 0.32, 0.35, 0.97, 0.45, 0.64],
    },
    "supply_chain_analyst": {
        "industries": [
            "manufacturing_logistics",
            "retail_ecommerce",
            "professional_services",
            "other",
        ],
        "current_roles": [
            "supply_logistics",
            "admin_operations",
            "quality_process",
        ],
        "experience": 5.0,
        "leadership": 1.0,
        "skills": [0.84, 0.57, 0.68, 0.66, 0.27, 0.51, 0.26, 0.97, 0.56],
    },
    "fpa_analyst": {
        "industries": [
            "financial_services",
            "professional_services",
            "retail_ecommerce",
            "manufacturing_logistics",
        ],
        "current_roles": [
            "finance_planning",
            "accounting",
            "data_reporting",
        ],
        "experience": 5.5,
        "leadership": 1.0,
        "skills": [0.87, 0.54, 0.67, 0.63, 0.25, 0.98, 0.28, 0.56, 0.43],
    },
    "customer_experience_manager": {
        "industries": [
            "it_bpo",
            "retail_ecommerce",
            "tourism_hospitality",
            "financial_services",
        ],
        "current_roles": [
            "customer_service",
            "team_lead_supervisor",
            "sales_business",
            "hospitality_tourism",
        ],
        "experience": 7.0,
        "leadership": 3.0,
        "skills": [0.52, 0.51, 0.95, 0.78, 0.50, 0.37, 0.55, 0.65, 0.98],
    },
    "ux_researcher": {
        "industries": [
            "it_bpo",
            "retail_ecommerce",
            "professional_services",
            "education",
        ],
        "current_roles": [
            "product_ux_design",
            "marketing_content",
            "education_training",
        ],
        "experience": 4.5,
        "leadership": 0.8,
        "skills": [0.75, 0.65, 0.94, 0.57, 0.93, 0.25, 0.45, 0.38, 0.88],
    },
}

LEADERSHIP_PATHWAYS: dict[str, dict[str, Any]] = {
    "data_bi_analyst": {
        "title": "Data & Business Intelligence Leadership Pathway",
        "summary": (
            "Build from analytics delivery toward ownership of decision systems, "
            "stakeholder alignment, data quality, and an analytics team or portfolio."
        ),
        "progression": (
            "Senior Data/BI Analyst → Analytics or BI Lead → Analytics/BI Manager"
        ),
        "focus": (
            "Move beyond producing dashboards: set metric definitions, prioritize "
            "analysis, coach reviewers, and hold decision-makers to a shared evidence base."
        ),
        "proof": (
            "Lead one cross-functional dashboard or metric-governance initiative. "
            "Document the decision, contributors, review process, adoption result, "
            "and what you changed after stakeholder feedback."
        ),
        "core_skills": ["data_analytics"],
    },
    "cybersecurity_analyst": {
        "title": "Cybersecurity Leadership Pathway",
        "summary": (
            "Progress from security analysis toward incident leadership, control "
            "ownership, risk communication, and accountable security operations."
        ),
        "progression": (
            "Senior Cybersecurity Analyst → Security Operations Lead → "
            "Cybersecurity Manager"
        ),
        "focus": (
            "Build judgment under pressure, escalation discipline, control ownership, "
            "and the ability to explain technical risk to business leaders."
        ),
        "proof": (
            "Lead a tabletop incident exercise or control-improvement workstream. "
            "Record the decision log, responsibilities, lessons learned, and verified "
            "follow-through without exposing confidential security information."
        ),
        "core_skills": ["technology"],
    },
    "cloud_solutions_engineer": {
        "title": "Cloud & Platform Engineering Leadership Pathway",
        "summary": (
            "Develop from technical delivery toward architecture decisions, service "
            "reliability, engineering standards, and team enablement."
        ),
        "progression": (
            "Senior Cloud Engineer → Cloud/Platform Technical Lead → "
            "Cloud or Platform Engineering Manager"
        ),
        "focus": (
            "Practice technical prioritization, architecture trade-offs, reliability "
            "ownership, mentoring, and clear communication with nontechnical stakeholders."
        ),
        "proof": (
            "Lead a small reliability, cost, or migration improvement. Publish a "
            "sanitized decision record covering alternatives, risks, contributors, "
            "measured results, and operational handoff."
        ),
        "core_skills": ["technology"],
    },
    "project_manager": {
        "title": "Project & Program Leadership Pathway",
        "summary": (
            "Move toward accountable ownership of outcomes, cross-functional delivery, "
            "risk decisions, and eventually a portfolio of related initiatives."
        ),
        "progression": (
            "Project or Business Analyst → Project Manager → Program Manager"
        ),
        "focus": (
            "Strengthen scope judgment, negotiation, risk escalation, benefits tracking, "
            "and the ability to align people who do not report directly to you."
        ),
        "proof": (
            "Own one bounded cross-functional initiative from charter through review. "
            "Show the decision log, risks, stakeholder agreements, outcome measures, "
            "and retrospective—not confidential project material."
        ),
        "core_skills": ["project_management", "communication"],
    },
    "digital_marketing_strategist": {
        "title": "Digital Marketing Leadership Pathway",
        "summary": (
            "Progress from campaign execution toward portfolio strategy, experimentation "
            "standards, commercial accountability, and creative-team direction."
        ),
        "progression": (
            "Senior Digital Strategist → Growth or Marketing Lead → "
            "Digital Marketing Manager"
        ),
        "focus": (
            "Develop prioritization across channels, responsible experimentation, "
            "budget judgment, coaching, and alignment with sales and customer teams."
        ),
        "proof": (
            "Lead a multi-channel experiment with a written hypothesis, budget guardrails, "
            "review roles, results, and a decision about what the team should stop, start, "
            "or scale."
        ),
        "core_skills": ["creative_design", "communication"],
    },
    "people_analytics_specialist": {
        "title": "People Analytics Leadership Pathway",
        "summary": (
            "Use workforce evidence as a foundation for leading responsible analytics, "
            "workforce-planning decisions, stakeholder trust, and a people-insights team."
        ),
        "progression": (
            "People Analytics Specialist → People Analytics Lead → "
            "People Analytics Manager"
        ),
        "focus": (
            "Build privacy judgment, ethical review, consulting skill, team coaching, "
            "and the discipline to separate correlation from causal claims."
        ),
        "proof": (
            "Lead an anonymized workforce-insight project with privacy safeguards, "
            "bias checks, stakeholder review, a decision log, and a clearly measured "
            "organizational outcome."
        ),
        "core_skills": ["people_hr", "data_analytics"],
    },
    "supply_chain_analyst": {
        "title": "Supply Chain & Operations Leadership Pathway",
        "summary": (
            "Advance from analysis toward planning decisions, service-level trade-offs, "
            "cross-functional operations, and accountable supply-chain leadership."
        ),
        "progression": (
            "Senior Supply Chain Analyst → Planning or Operations Lead → "
            "Supply Chain Manager"
        ),
        "focus": (
            "Strengthen scenario judgment, supplier and stakeholder communication, "
            "operating cadence, risk ownership, and continuous-improvement leadership."
        ),
        "proof": (
            "Lead one inventory, service-level, or process-improvement cycle. Show the "
            "baseline, trade-offs, contributors, decision, measured result, and control "
            "used to sustain the change."
        ),
        "core_skills": ["operations"],
    },
    "fpa_analyst": {
        "title": "FP&A Leadership Pathway",
        "summary": (
            "Build from forecasting and analysis toward business partnership, scenario "
            "ownership, management challenge, and leadership of a planning function."
        ),
        "progression": (
            "Senior FP&A Analyst or Finance Business Partner → FP&A Lead → FP&A Manager"
        ),
        "focus": (
            "Develop commercial judgment, constructive challenge, planning governance, "
            "executive communication, coaching, and accountability for assumptions."
        ),
        "proof": (
            "Lead a driver-based planning or variance-review cycle. Document assumptions, "
            "contributors, control checks, recommendations, leadership decisions, and "
            "the result of the chosen action."
        ),
        "core_skills": ["finance", "data_analytics"],
    },
    "customer_experience_manager": {
        "title": "Customer Experience Leadership Pathway",
        "summary": (
            "Lead customer evidence, service design, operating improvements, and the "
            "human–AI handoff across customer-facing teams."
        ),
        "progression": (
            "Customer Experience Specialist or Team Lead → Customer Experience Manager "
            "→ Head of Customer Experience"
        ),
        "focus": (
            "Strengthen journey-level prioritization, service recovery, coaching, "
            "cross-functional influence, and accountability for customer outcomes."
        ),
        "proof": (
            "Lead one customer-journey improvement from evidence to implementation. "
            "Show the research, contributors, decision, operating change, customer "
            "measure, and follow-up review."
        ),
        "core_skills": ["customer_experience", "communication"],
    },
    "ux_researcher": {
        "title": "UX Research Leadership Pathway",
        "summary": (
            "Progress from individual research delivery toward research strategy, "
            "quality standards, stakeholder influence, and team development."
        ),
        "progression": (
            "Senior UX Researcher → UX Research Lead → UX Research Manager"
        ),
        "focus": (
            "Build research-program prioritization, ethical practice, coaching, "
            "cross-functional influence, and a consistent standard for evidence quality."
        ),
        "proof": (
            "Lead a multi-stakeholder research initiative with a documented research "
            "decision, participant safeguards, review roles, product response, and "
            "evidence of what changed."
        ),
        "core_skills": [
            "creative_design",
            "communication",
            "customer_experience",
        ],
    },
}


def _clip_present_rating(value: float) -> int:
    return int(np.clip(np.rint(value), 1, 5))


def generate_synthetic_profiles(
    seed: int = SEED,
    per_career: int = SYNTHETIC_PROFILES_PER_CAREER,
) -> pd.DataFrame:
    """Generate a larger, overlapping teaching set from documented assumptions."""

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    profile_number = 1
    career_ids = list(PROTOTYPES)
    role_ids = list(CURRENT_ROLES)

    for career_index, (career_id, prototype) in enumerate(PROTOTYPES.items()):
        adjacent = PROTOTYPES[career_ids[(career_index + 1) % len(career_ids)]]
        for _ in range(per_career):
            industry = (
                rng.choice(prototype["industries"])
                if rng.random() < 0.76
                else rng.choice(list(INDUSTRIES))
            )
            role_draw = rng.random()
            if role_draw < 0.76:
                current_role = str(rng.choice(prototype["current_roles"]))
            elif role_draw < 0.92:
                current_role = str(rng.choice(adjacent["current_roles"]))
            else:
                current_role = str(rng.choice(role_ids))
            current_job_title = str(
                rng.choice(CURRENT_ROLES[current_role]["titles"])
            )

            # A minority of profiles blend with a neighboring pathway. This
            # creates more realistic overlap than ten perfectly separated clusters.
            blend = float(rng.uniform(0.08, 0.28)) if rng.random() < 0.30 else 0.0
            experience_center = (
                prototype["experience"] * (1 - blend)
                + adjacent["experience"] * blend
            )
            leadership_center = (
                prototype["leadership"] * (1 - blend)
                + adjacent["leadership"] * blend
            )
            years = float(
                np.clip(rng.normal(experience_center, 2.0), 0, 25)
            )
            leadership = float(
                np.clip(
                    rng.normal(leadership_center, 1.0),
                    0,
                    min(15, years),
                )
            )
            row: dict[str, Any] = {
                "profile_id": f"PH-{profile_number:03d}",
                "current_industry": str(industry),
                "current_role": current_role,
                "current_job_title": current_job_title,
                "years_experience": round(years, 1),
                "leadership_years": round(leadership, 1),
            }
            for skill_index, (skill, center) in enumerate(
                zip(SKILLS, prototype["skills"])
            ):
                mixed_center = (
                    center * (1 - blend)
                    + adjacent["skills"][skill_index] * blend
                )
                # Zero represents no experience with the skill. It is more
                # common for low-adjacency skills, while every pathway can
                # still contain a small number of genuine beginners.
                no_experience_probability = (
                    0.03 + 0.20 * (1 - mixed_center) ** 2
                )
                row[skill] = (
                    0
                    if rng.random() < no_experience_probability
                    else _clip_present_rating(
                        1 + 4 * mixed_center + rng.normal(0, 0.58)
                    )
                )
            row["recommended_career"] = career_id
            rows.append(row)
            profile_number += 1

    frame = pd.DataFrame(rows)
    expected_rows = len(PROTOTYPES) * per_career
    if (
        len(frame) != expected_rows
        or not frame["recommended_career"].value_counts().eq(per_career).all()
    ):
        raise RuntimeError("Synthetic data integrity check failed.")
    return frame


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def leadership_readiness_for(
    profile: dict[str, Any],
    skill_fit: float,
) -> float:
    """Return a transparent, synthetic leadership-readiness index."""

    experience_signal = min(float(profile["years_experience"]) / 8, 1.0)
    formal_leadership_signal = min(float(profile["leadership_years"]) / 3, 1.0)
    transferable_signal = (
        0.40 * float(profile["communication"]) / 5
        + 0.35 * float(profile["project_management"]) / 5
        + 0.25 * float(profile["people_hr"]) / 5
    )
    general_readiness = (
        0.20 * experience_signal
        + 0.45 * formal_leadership_signal
        + 0.35 * transferable_signal
    )
    return float(np.clip(0.75 * general_readiness + 0.25 * skill_fit, 0, 1))


def leadership_stage_for(
    profile: dict[str, Any],
    readiness: float,
) -> dict[str, str]:
    """Describe an investigation stage without claiming promotion readiness."""

    leadership_years = float(profile["leadership_years"])
    if leadership_years >= 3 and readiness >= 0.72:
        return {
            "label": "Investigate manager-level opportunities",
            "insight": (
                "Your profile shows sustained leadership exposure and stronger "
                "transferable signals. Validate scope, people responsibility, and "
                "decision authority against current job descriptions and references."
            ),
        }
    if leadership_years >= 1 and readiness >= 0.55:
        return {
            "label": "Investigate lead or senior-lead opportunities",
            "insight": (
                "Your profile suggests a bridge through workstream ownership, mentoring, "
                "or a formal lead role before assuming full manager accountability."
            ),
        }
    return {
        "label": "Build a leadership bridge first",
        "insight": (
            "The selected direction is leadership, but the profile does not yet show "
            "enough sustained leadership evidence for this prototype to imply immediate "
            "manager readiness. Build proof through a bounded initiative, mentoring, "
            "decision ownership, and measurable team or stakeholder outcomes."
        ),
    }


def leadership_gaps_for(
    profile: dict[str, Any],
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Prioritize transparent leadership-building targets for this teaching tool."""

    targets = {
        "communication": 4,
        "project_management": 4,
        "people_hr": 3,
        "operations": 3,
    }
    gaps = [
        {
            "skill": skill,
            "label": SKILLS[skill]["label"],
            "current": float(profile[skill]),
            "target": target,
            "gap": target - float(profile[skill]),
        }
        for skill, target in targets.items()
        if target - float(profile[skill]) > 0
    ]
    return sorted(gaps, key=lambda item: item["gap"], reverse=True)[:limit]


@dataclass
class CareerEngine:
    model: Pipeline
    synthetic_profiles: pd.DataFrame

    @classmethod
    def train(cls) -> "CareerEngine":
        profiles = generate_synthetic_profiles()
        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", StandardScaler(), NUMERIC_FEATURES),
                ("industry", _one_hot_encoder(), CATEGORICAL_FEATURES),
                (
                    "job_title",
                    TfidfVectorizer(
                        lowercase=True,
                        ngram_range=(1, 2),
                        min_df=2,
                        strip_accents="unicode",
                    ),
                    TEXT_FEATURE,
                ),
            ],
            remainder="drop",
        )
        model = Pipeline(
            [
                ("preprocess", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        class_weight="balanced_subsample",
                        random_state=SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        features = profiles[
            CATEGORICAL_FEATURES + NUMERIC_FEATURES + [TEXT_FEATURE]
        ]
        model.fit(features, profiles["recommended_career"])
        return cls(model=model, synthetic_profiles=profiles)

    def recommend(
        self,
        profile: dict[str, Any],
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        validate_profile(profile)
        input_frame = pd.DataFrame([profile])[
            CATEGORICAL_FEATURES + NUMERIC_FEATURES + [TEXT_FEATURE]
        ]
        probabilities = self.model.predict_proba(input_frame)[0]
        class_ids = self.model.named_steps["model"].classes_
        results: list[dict[str, Any]] = []

        for career_id, model_probability in zip(class_ids, probabilities):
            career = CAREERS[str(career_id)]
            current = career["current_demand"]
            future = career["future_demand"]
            targets = target_ratings(str(career_id))
            skill_fit = 1 - np.mean(
                [
                    abs(float(profile[skill]) - target) / 5
                    for skill, target in targets.items()
                ]
            )
            experience_fit = min(
                1.0,
                float(profile["years_experience"])
                / max(career["min_experience"], 1),
            )
            industry_fit = (
                1.0
                if profile["current_industry"] in career["preferred_industries"]
                else 0.55
            )
            role_fit = (
                1.0
                if profile["current_role"]
                in PROTOTYPES[str(career_id)]["current_roles"]
                else 0.35
            )

            is_leadership_goal = profile["goal"] == "leadership"
            leadership_readiness = leadership_readiness_for(profile, skill_fit)
            leadership_pathway_definition = LEADERSHIP_PATHWAYS[str(career_id)]
            core_skill_fit = float(
                np.mean(
                    [
                        float(profile[skill]) / 5
                        for skill in leadership_pathway_definition["core_skills"]
                    ]
                )
            )
            career_direction_fit = (
                0.35 * role_fit
                + 0.15 * industry_fit
                + 0.20 * float(skill_fit)
                + 0.20 * core_skill_fit
                + 0.10 * leadership_readiness
            )

            goal_adjustment = 0.0
            if profile["goal"] == "future_ready":
                goal_adjustment = max(
                    0.0,
                    future["score"] - current["score"],
                ) * 0.03
            elif profile["goal"] == "sector_switch" and industry_fit < 1:
                goal_adjustment = future["score"] * 0.03

            if is_leadership_goal:
                # Leadership is a direction-selection policy, not a synthetic
                # promotion prediction. Career-direction fit carries 30% so the
                # selected goal materially affects ranking and presentation.
                score = 100 * min(
                    1.0,
                    0.25 * float(model_probability)
                    + 0.15 * float(skill_fit)
                    + 0.10 * role_fit
                    + 0.06 * experience_fit
                    + 0.04 * current["score"]
                    + 0.06 * future["score"]
                    + 0.04 * industry_fit
                    + 0.30 * career_direction_fit,
                )
            else:
                score = 100 * min(
                    1.0,
                    0.42 * float(model_probability)
                    + 0.20 * float(skill_fit)
                    + 0.05 * experience_fit
                    + 0.14 * role_fit
                    + 0.05 * current["score"]
                    + 0.07 * future["score"]
                    + 0.04 * industry_fit
                    + goal_adjustment,
                )

            leadership_pathway = (
                leadership_pathway_definition
                if is_leadership_goal
                else None
            )
            results.append(
                {
                    "career_id": str(career_id),
                    "career": career["title"],
                    "summary": career["summary"],
                    "display_career": (
                        leadership_pathway["title"]
                        if leadership_pathway
                        else career["title"]
                    ),
                    "display_summary": (
                        leadership_pathway["summary"]
                        if leadership_pathway
                        else career["summary"]
                    ),
                    "recommendation_score": round(score, 1),
                    "synthetic_model_fit": round(
                        100 * float(model_probability),
                        1,
                    ),
                    "skill_fit": round(100 * float(skill_fit), 1),
                    "experience_fit": round(100 * experience_fit, 1),
                    "industry_fit": round(100 * industry_fit, 1),
                    "role_fit": round(100 * role_fit, 1),
                    "career_direction_fit": round(
                        100 * career_direction_fit,
                        1,
                    ),
                    "core_skill_fit": round(100 * core_skill_fit, 1),
                    "leadership_readiness": round(
                        100 * leadership_readiness,
                        1,
                    ),
                    "leadership_stage": (
                        leadership_stage_for(profile, leadership_readiness)
                        if is_leadership_goal
                        else None
                    ),
                    "leadership_pathway": leadership_pathway,
                    "leadership_gaps": (
                        leadership_gaps_for(profile)
                        if is_leadership_goal
                        else []
                    ),
                    "goal": profile["goal"],
                    "skill_gaps": skill_gaps_for(profile, str(career_id)),
                    **career,
                }
            )

        return sorted(
            results,
            key=lambda item: (
                item["recommendation_score"],
                item["synthetic_model_fit"],
            ),
            reverse=True,
        )[:top_k]


def validate_profile(profile: dict[str, Any]) -> None:
    expected = CATEGORICAL_FEATURES + NUMERIC_FEATURES + [TEXT_FEATURE, "goal"]
    missing = [field for field in expected if field not in profile]
    if missing:
        raise ValueError(f"Missing profile fields: {', '.join(missing)}")
    if profile["current_industry"] not in INDUSTRIES:
        raise ValueError("Please select a listed industry.")
    if profile["current_role"] not in CURRENT_ROLES:
        raise ValueError("Please select a listed current-role family.")
    if len(str(profile["current_job_title"]).strip()) < 2:
        raise ValueError("Please enter your current job title.")
    if profile["goal"] not in CAREER_GOALS:
        raise ValueError("Please select a listed career goal.")
    years = float(profile["years_experience"])
    leadership = float(profile["leadership_years"])
    if not 0 <= years <= 50:
        raise ValueError("Years of experience must be between 0 and 50.")
    if not 0 <= leadership <= years:
        raise ValueError(
            "Years leading others cannot be greater than total work experience."
        )
    for skill in SKILLS:
        if not 0 <= float(profile[skill]) <= 5:
            raise ValueError(f"{SKILLS[skill]['label']} must be between 0 and 5.")


def target_ratings(career_id: str) -> dict[str, int]:
    return {
        skill: int(np.clip(np.rint(1 + 4 * center), 1, 5))
        for skill, center in zip(SKILLS, PROTOTYPES[career_id]["skills"])
    }


def skill_gaps_for(
    profile: dict[str, Any],
    career_id: str,
    limit: int = 3,
) -> list[dict[str, Any]]:
    targets = target_ratings(career_id)
    gaps = [
        {
            "skill": skill,
            "label": SKILLS[skill]["label"],
            "current": float(profile[skill]),
            "target": target,
            "gap": target - float(profile[skill]),
        }
        for skill, target in targets.items()
        if target - float(profile[skill]) > 0
    ]
    return sorted(gaps, key=lambda item: item["gap"], reverse=True)[:limit]
