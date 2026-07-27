"""Skills-first career recommendation engine for the Hakbang PH app.

The model is trained only on deterministic synthetic skill profiles. Industry,
employer, current title, demographics, and protected characteristics are not model
features. Runtime factual content comes only from the fixed, source-linked registry;
no generative AI creates career, demand, or credential claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 20260725
SYNTHETIC_PROFILES_PER_CAREER = 200
DATASET_VERSION = "PH-SKILLS-FIRST-SYN-2200-2026.07.28"
EVIDENCE_CHECKED = "28 July 2026"
MODEL_NAME = "Extra Trees"
SYNTHETIC_CV_MACRO_F1 = 0.7854
SYNTHETIC_CV_ACCURACY = 0.7868

# Retained only as a migration reference for profiles created before the
# skills-first v5 schema. None of these legacy fields enter model training or
# recommendation scoring.
LEGACY_INDUSTRIES = {
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

LEGACY_CURRENT_ROLES = {
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
    "science_laboratory": {
        "label": "Science, chemistry, or laboratory practice",
        "titles": [
            "Chemist",
            "Quality Control Chemist",
            "Laboratory Analyst",
            "Research Scientist",
            "Laboratory Scientist",
            "Chemical Technician",
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
        "hint": "Spreadsheets, reporting, SQL, statistics, dashboards, and evidence",
    },
    "ai_automation": {
        "label": "AI & automation",
        "hint": "Responsible AI use, prompting, workflow automation, and output checking",
    },
    "software_cloud": {
        "label": "Software & cloud",
        "hint": "Applications, systems, cloud platforms, coding, and troubleshooting",
    },
    "cybersecurity_risk": {
        "label": "Cybersecurity & risk",
        "hint": "Security controls, privacy, governance, audit, and risk response",
    },
    "communication": {
        "label": "Communication & storytelling",
        "hint": "Writing, presenting, negotiation, facilitation, and explaining decisions",
    },
    "project_change": {
        "label": "Project & change delivery",
        "hint": "Planning, scope, risk, coordination, adoption, and improvement delivery",
    },
    "creative_design": {
        "label": "Creative & design",
        "hint": "Design, campaigns, prototyping, content, and ideation",
    },
    "finance_commercial": {
        "label": "Finance & commercial",
        "hint": "Budgeting, accounting, forecasting, pricing, and commercial analysis",
    },
    "people_coaching": {
        "label": "People & coaching",
        "hint": "Coaching, talent, teamwork, workforce planning, and employee experience",
    },
    "operations_quality": {
        "label": "Operations & quality",
        "hint": "Process improvement, logistics, controls, quality, and service delivery",
    },
    "customer_research": {
        "label": "Customer & user research",
        "hint": "Research, service design, customer success, usability, and voice of customer",
    },
    "scientific_laboratory": {
        "label": "Scientific & laboratory practice",
        "hint": (
            "Experimental methods, analytical chemistry, laboratory quality, "
            "validation, and safety"
        ),
    },
}

NUMERIC_FEATURES = [
    "years_experience",
    *SKILLS.keys(),
]
TEXT_FEATURE = None
CATEGORICAL_FEATURES: list[str] = []
LEGACY_CAREER_GOALS = {
    "future_ready": "Build skills for an AI-shaped future",
    "leadership": "Move toward leadership",
    "sector_switch": "Switch industry or sector",
    "build_specialty": "Deepen a specialist skill",
}

LEGACY_TITLE_ROLE_HINTS = {
    "chemist": "science_laboratory",
    "chemical technician": "science_laboratory",
    "laboratory analyst": "science_laboratory",
    "lab analyst": "science_laboratory",
    "laboratory scientist": "science_laboratory",
    "research scientist": "science_laboratory",
    "microbiologist": "science_laboratory",
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
    "prc_chem_lab": {
        "name": "Chemistry Law requirements for chemical laboratories",
        "owner": "Philippine Professional Regulation Commission",
        "url": (
            "https://prc.gov.ph/article/announcement-requirement-chemistry-law-"
            "certificate-authority-operate-chemical-laboratories"
        ),
        "published": "29 May 2017",
    },
    "iso_17025": {
        "name": "ISO/IEC 17025:2017 — Testing and calibration laboratories",
        "owner": "International Organization for Standardization",
        "url": "https://www.iso.org/standard/66912.html",
        "published": "Confirmed current in 2023; checked 28 July 2026",
    },
    "onet_bi": {
        "name": "Business Intelligence Analysts — occupation profile",
        "owner": "O*NET OnLine, U.S. Department of Labor",
        "url": "https://www.onetonline.org/link/summary/15-2051.01",
        "published": "Updated 2026; checked 28 July 2026",
    },
    "onet_cyber": {
        "name": "Information Security Analysts — occupation profile",
        "owner": "O*NET OnLine, U.S. Department of Labor",
        "url": "https://www.onetonline.org/link/summary/15-1212.00",
        "published": "Updated 2026; checked 28 July 2026",
    },
    "onet_systems": {
        "name": "Computer Systems Engineers/Architects — occupation profile",
        "owner": "O*NET OnLine, U.S. Department of Labor",
        "url": "https://www.onetonline.org/link/summary/15-1299.08",
        "published": "Updated 2026; checked 28 July 2026",
    },
    "onet_project": {
        "name": "Project Management Specialists — occupation profile",
        "owner": "O*NET OnLine, U.S. Department of Labor",
        "url": "https://www.onetonline.org/link/summary/13-1082.00",
        "published": "Updated 2026; checked 28 July 2026",
    },
    "onet_market": {
        "name": "Market Research Analysts and Marketing Specialists — occupation profile",
        "owner": "O*NET OnLine, U.S. Department of Labor",
        "url": "https://www.onetonline.org/link/summary/13-1161.00",
        "published": "Updated 2026; checked 28 July 2026",
    },
    "onet_hr": {
        "name": "Human Resources Specialists — occupation profile",
        "owner": "O*NET OnLine, U.S. Department of Labor",
        "url": "https://www.onetonline.org/link/summary/13-1071.00",
        "published": "Updated 2026; checked 28 July 2026",
    },
    "onet_logistics": {
        "name": "Logisticians — occupation profile",
        "owner": "O*NET OnLine, U.S. Department of Labor",
        "url": "https://www.onetonline.org/link/summary/13-1081.00",
        "published": "Updated 2026; checked 28 July 2026",
    },
    "onet_finance": {
        "name": "Financial and Investment Analysts — occupation profile",
        "owner": "O*NET OnLine, U.S. Department of Labor",
        "url": "https://www.onetonline.org/link/summary/13-2051.00",
        "published": "Updated 2026; checked 28 July 2026",
    },
    "onet_quality": {
        "name": "Quality Control Analysts — occupation profile",
        "owner": "O*NET OnLine, U.S. Department of Labor",
        "url": "https://www.onetonline.org/link/summary/19-4099.01",
        "published": "Updated 2026; checked 28 July 2026",
    },
    "onet_operations": {
        "name": "General and Operations Managers — occupation profile",
        "owner": "O*NET OnLine, U.S. Department of Labor",
        "url": "https://www.onetonline.org/link/summary/11-1021.00",
        "published": "Updated 2026; checked 28 July 2026",
    },
    "onet_digital_design": {
        "name": "Web and Digital Interface Designers — occupation profile",
        "owner": "O*NET OnLine, U.S. Department of Labor",
        "url": "https://www.onetonline.org/link/summary/15-1255.00",
        "published": "Updated 2026; checked 28 July 2026",
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
    "laboratory_quality_specialist": {
        "title": "Laboratory Quality & Scientific Operations Specialist",
        "summary": (
            "Strengthen laboratory quality, method reliability, safety, compliance, "
            "and the operating systems that support defensible scientific results."
        ),
        "min_experience": 2,
        "preferred_industries": [
            "healthcare",
            "manufacturing_logistics",
            "government_public",
            "professional_services",
        ],
        "current_demand": demand(
            "Role-relevant",
            0.7,
            "Philippine regulatory evidence—not a vacancy count",
            "PRC states that only registered chemists can head a chemical analyses "
            "laboratory, supervise chemical work, and certify analyses in covered "
            "Philippine laboratories. This directly supports a regulated leadership "
            "path, but it does not quantify available jobs.",
            ["prc_chem_lab"],
        ),
        "future_demand": demand(
            "Strong standards relevance",
            0.8,
            "Standards-based directional evidence—not an employment forecast",
            "ISO/IEC 17025 remains the international competence standard for testing "
            "and calibration laboratories. ISO notes its focus on competence, "
            "impartiality, consistent operation, information technology, and senior "
            "management responsibility. This supports continued quality-leadership "
            "relevance without proving future vacancy growth.",
            ["iso_17025"],
        ),
        "ai_opportunity": (
            "Use automation and AI to organize controlled documents, review structured "
            "quality data, detect trends, and draft investigation questions while "
            "keeping method validation, traceability, uncertainty, safety, and final "
            "scientific judgment under qualified human control."
        ),
        "human_edge": (
            "Scientific accountability, laboratory safety, method suitability, "
            "traceability, ethical escalation, and responsibility for defensible results."
        ),
        "first_proof": (
            "Create a sanitized laboratory-improvement case: map one controlled process, "
            "identify a quality or safety risk, propose an evidence-based change, and "
            "define verification, documentation, and escalation controls."
        ),
        "certification": {
            "name": "Certified Quality Improvement Associate (CQIA)",
            "issuer": "American Society for Quality (ASQ)",
            "url": "https://www.asq.org/cert/quality-improvement-associate",
            "eligibility": (
                "ASQ currently requires two years of full-time paid work experience, "
                "or an associate degree or two years of equivalent higher education. "
                "Confirm the current requirements and exam availability with ASQ."
            ),
            "why_it_fits": (
                "CQIA covers foundational quality tools, improvement methods, and "
                "teamwork. It can complement chemistry expertise when building evidence "
                "for laboratory-quality and process-improvement responsibilities."
            ),
            "practitioner": (
                "Krystel Sherman, ASQ CQIA · quality-control microbiologist and chemist"
            ),
            "practitioner_insight": (
                "Her public professional profile lists CQIA alongside a chemistry and "
                "quality-control career. The available profile establishes credential "
                "use in a relevant profession but does not claim that CQIA caused a "
                "promotion or employment outcome."
            ),
            "source_type": (
                "Public credential-holder profile; no causal career outcome reported"
            ),
            "practitioner_url": (
                "https://www.linkedin.com/in/krystel-sherman-asq-cqia-at-50235711"
            ),
            "caveat": (
                "CQIA is a foundational quality credential, not a chemistry license, "
                "ISO/IEC 17025 laboratory accreditation, or proof of leadership. "
                "Philippine chemistry practice and laboratory-head responsibilities "
                "remain subject to PRC registration and applicable law."
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

# Legacy role-family prototypes are retained only as an audit trail for the
# pre-v5 model. The deployed skills-first model uses JOB_PROTOTYPES below.
LEGACY_PROTOTYPES = {
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
        "skills": [
            0.92, 0.78, 0.66, 0.53, 0.34, 0.56, 0.31, 0.52, 0.48, 0.25
        ],
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
        "skills": [
            0.60, 0.96, 0.62, 0.55, 0.20, 0.30, 0.20, 0.54, 0.37, 0.20
        ],
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
        "skills": [
            0.66, 0.97, 0.66, 0.68, 0.25, 0.28, 0.20, 0.63, 0.42, 0.15
        ],
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
        "skills": [
            0.58, 0.57, 0.92, 0.97, 0.35, 0.44, 0.55, 0.69, 0.72, 0.20
        ],
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
        "skills": [
            0.72, 0.62, 0.88, 0.61, 0.96, 0.35, 0.28, 0.42, 0.82, 0.10
        ],
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
        "skills": [
            0.84, 0.58, 0.86, 0.58, 0.32, 0.35, 0.97, 0.45, 0.64, 0.25
        ],
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
        "skills": [
            0.84, 0.57, 0.68, 0.66, 0.27, 0.51, 0.26, 0.97, 0.56, 0.30
        ],
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
        "skills": [
            0.87, 0.54, 0.67, 0.63, 0.25, 0.98, 0.28, 0.56, 0.43, 0.25
        ],
    },
    "laboratory_quality_specialist": {
        "industries": [
            "healthcare",
            "manufacturing_logistics",
            "government_public",
            "professional_services",
        ],
        "current_roles": [
            "science_laboratory",
            "quality_process",
        ],
        "experience": 4.5,
        "leadership": 1.0,
        "skills": [
            0.55, 0.45, 0.65, 0.55, 0.20, 0.20, 0.30, 0.75, 0.25, 0.98
        ],
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
        "skills": [
            0.52, 0.51, 0.95, 0.78, 0.50, 0.37, 0.55, 0.65, 0.98, 0.15
        ],
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
        "skills": [
            0.75, 0.65, 0.94, 0.57, 0.93, 0.25, 0.45, 0.38, 0.88, 0.35
        ],
    },
}


# Each vector follows SKILLS insertion order and is a documented design
# assumption on a 0–1 scale. These are not employer requirements or occupational
# standards. They generate a balanced teaching dataset and make the ranking logic
# inspectable.
JOB_PROTOTYPES: dict[str, dict[str, Any]] = {
    "data_bi_analyst": {
        "experience": 4.0,
        "skills": [0.98, 0.78, 0.60, 0.35, 0.72, 0.60, 0.25, 0.55, 0.25, 0.50, 0.45, 0.20],
        "core_skills": ["data_analytics"],
        "application_contexts": [
            "Finance and insurance",
            "Healthcare and life sciences",
            "Retail and e-commerce",
            "Government and public services",
            "Technology and professional services",
        ],
    },
    "cybersecurity_analyst": {
        "experience": 3.5,
        "skills": [0.65, 0.70, 0.90, 0.98, 0.65, 0.65, 0.20, 0.25, 0.25, 0.70, 0.30, 0.15],
        "core_skills": ["cybersecurity_risk", "software_cloud"],
        "application_contexts": [
            "Banking and financial services",
            "Government and critical infrastructure",
            "Healthcare",
            "Telecommunications",
            "Technology and business-process services",
        ],
    },
    "cloud_solutions_engineer": {
        "experience": 4.5,
        "skills": [0.60, 0.80, 0.98, 0.75, 0.65, 0.72, 0.25, 0.25, 0.25, 0.75, 0.30, 0.15],
        "core_skills": ["software_cloud", "ai_automation"],
        "application_contexts": [
            "Technology and telecommunications",
            "Banking and financial services",
            "Retail and e-commerce",
            "Government digital services",
            "Professional and managed services",
        ],
    },
    "project_manager": {
        "experience": 6.5,
        "skills": [0.50, 0.60, 0.50, 0.35, 0.95, 0.98, 0.50, 0.50, 0.78, 0.82, 0.70, 0.15],
        "core_skills": ["project_change", "communication"],
        "application_contexts": [
            "Technology transformation",
            "Construction and infrastructure",
            "Healthcare",
            "Finance and professional services",
            "Government and nonprofit programs",
        ],
    },
    "digital_marketing_strategist": {
        "experience": 3.5,
        "skills": [0.68, 0.72, 0.58, 0.30, 0.88, 0.68, 0.96, 0.32, 0.42, 0.50, 0.90, 0.10],
        "core_skills": ["creative_design", "communication", "customer_research"],
        "application_contexts": [
            "Retail and e-commerce",
            "Consumer goods",
            "Tourism and hospitality",
            "Technology and media",
            "Education and professional services",
        ],
    },
    "people_analytics_specialist": {
        "experience": 4.0,
        "skills": [0.85, 0.65, 0.55, 0.42, 0.82, 0.68, 0.40, 0.38, 0.98, 0.55, 0.60, 0.15],
        "core_skills": ["people_coaching", "data_analytics"],
        "application_contexts": [
            "Large multi-industry employers",
            "Business-process services",
            "Finance and professional services",
            "Healthcare",
            "Government and education",
        ],
    },
    "supply_chain_analyst": {
        "experience": 4.0,
        "skills": [0.84, 0.68, 0.58, 0.30, 0.68, 0.72, 0.25, 0.58, 0.35, 0.98, 0.55, 0.20],
        "core_skills": ["operations_quality", "data_analytics"],
        "application_contexts": [
            "Manufacturing",
            "Retail and e-commerce",
            "Transport and logistics",
            "Healthcare supply networks",
            "Food, agriculture, and consumer goods",
        ],
    },
    "fpa_analyst": {
        "experience": 4.5,
        "skills": [0.90, 0.70, 0.52, 0.40, 0.75, 0.70, 0.25, 0.99, 0.32, 0.66, 0.42, 0.20],
        "core_skills": ["finance_commercial", "data_analytics"],
        "application_contexts": [
            "Finance and insurance",
            "Retail and consumer goods",
            "Manufacturing",
            "Technology and telecommunications",
            "Professional services",
        ],
    },
    "laboratory_quality_specialist": {
        "experience": 3.5,
        "skills": [0.62, 0.55, 0.40, 0.45, 0.68, 0.58, 0.20, 0.18, 0.30, 0.85, 0.25, 0.99],
        "core_skills": ["scientific_laboratory", "operations_quality"],
        "application_contexts": [
            "Pharmaceutical and healthcare laboratories",
            "Food and beverage testing",
            "Chemicals and manufacturing",
            "Environmental and government laboratories",
            "Research and testing services",
        ],
    },
    "customer_experience_manager": {
        "experience": 5.5,
        "skills": [0.55, 0.65, 0.45, 0.32, 0.97, 0.80, 0.65, 0.35, 0.85, 0.72, 0.99, 0.15],
        "core_skills": ["customer_research", "communication"],
        "application_contexts": [
            "Business-process and contact-center services",
            "Retail and e-commerce",
            "Banking and insurance",
            "Telecommunications",
            "Travel, hospitality, and healthcare",
        ],
    },
    "ux_researcher": {
        "experience": 3.5,
        "skills": [0.65, 0.72, 0.58, 0.30, 0.92, 0.66, 0.98, 0.25, 0.70, 0.42, 0.98, 0.20],
        "core_skills": ["customer_research", "creative_design", "communication"],
        "application_contexts": [
            "Software and digital products",
            "Finance and fintech",
            "Retail and e-commerce",
            "Healthcare technology",
            "Government and public digital services",
        ],
    },
}


def skill_reason(
    why: str,
    example: str,
    sources: list[str],
) -> dict[str, Any]:
    """Create a fixed, source-linked explanation for one role skill."""
    return {
        "why": why,
        "example": example,
        "sources": sources,
    }


# Plain-language guidance is deliberately fixed rather than model-generated.
# O*NET is used as an occupational task reference, not as a Philippine licensing
# standard or a claim about one employer's vacancy requirements.
ROLE_GUIDANCE: dict[str, dict[str, Any]] = {
    "data_bi_analyst": {
        "fit": (
            "This job is a close match when you can work with data and explain what "
            "the numbers mean. The job is not only about making charts. It involves "
            "checking the data, finding useful patterns, and turning those patterns "
            "into information that managers or teams can use."
        ),
        "typical_work": [
            "Build and maintain reports, dashboards, or other decision tools.",
            "Check whether data is complete, consistent, and suitable for the question.",
            "Explain trends and recommend a practical action to a non-technical audience.",
        ],
        "skill_reasons": {
            "data_analytics": skill_reason(
                "BI analysts query data, prepare reports, and identify patterns or trends. "
                "Without this skill, it is difficult to tell whether a result is real, "
                "relevant, or caused by poor-quality data.",
                "Clean a sales or operations dataset, calculate useful measures, and build "
                "a dashboard that answers a specific business question.",
                ["onet_bi"],
            ),
            "communication": skill_reason(
                "The analysis only creates value when other people understand it. O*NET "
                "lists interpreting information for others and reporting to managers as "
                "part of BI work.",
                "Explain one important trend in plain language, state what is uncertain, "
                "and recommend the next decision or investigation.",
                ["onet_bi"],
            ),
            "software_cloud": skill_reason(
                "BI work uses databases, spreadsheets, reporting platforms, and dashboard "
                "tools. Technical confidence is needed to retrieve data and maintain the "
                "tools that other people rely on.",
                "Use SQL or a reporting tool to connect a data source, document the logic, "
                "and make the report repeatable.",
                ["onet_bi"],
            ),
            "ai_automation": skill_reason(
                "AI and automation can speed up repetitive analysis, drafting, and data "
                "preparation. The analyst still needs enough knowledge to check the output "
                "and stop errors from reaching a decision-maker.",
                "Automate one repeatable reporting step, then add checks for missing data, "
                "unexpected values, and incorrect calculations.",
                ["wef_skills", "ilo_genai", "onet_bi"],
            ),
        },
        "ai_explanation": (
            "AI can help write a first query, summarize a table, or draft a chart "
            "description. It does not know whether the source data is trustworthy or "
            "whether a pattern makes sense for the organization. A competitive analyst "
            "uses AI for speed but keeps responsibility for the question, the checks, "
            "and the recommendation."
        ),
        "ai_actions": [
            "Learn to use an AI assistant without uploading confidential business data.",
            "Check AI-produced calculations, code, labels, and conclusions against the source.",
            "Build judgment by explaining why a measure is useful and what it cannot prove.",
        ],
        "role_sources": ["onet_bi"],
        "ai_sources": ["wef_skills", "ilo_genai", "onet_bi"],
    },
    "cybersecurity_analyst": {
        "fit": (
            "This job is a close match when you can understand digital systems, notice "
            "risk, and follow evidence carefully. Security analysts protect information "
            "and systems by monitoring activity, finding weaknesses, documenting what "
            "happened, and helping people reduce the chance of another incident."
        ),
        "typical_work": [
            "Monitor systems or alerts and investigate unusual activity.",
            "Assess vulnerabilities and recommend controls that reduce risk.",
            "Document incidents, procedures, tests, and lessons for technical and business teams.",
        ],
        "skill_reasons": {
            "cybersecurity_risk": skill_reason(
                "The central responsibility is to identify vulnerabilities, assess risk, "
                "apply controls, and respond to security problems.",
                "Review a simple system, identify its most important risks, and propose "
                "controls with a clear reason for each one.",
                ["onet_cyber"],
            ),
            "software_cloud": skill_reason(
                "Security controls protect real networks, applications, accounts, and "
                "cloud services. Analysts need to understand how those systems normally "
                "work before they can recognize unsafe behavior.",
                "Create a small practice environment and document normal access, logging, "
                "patching, and account-permission settings.",
                ["onet_cyber"],
            ),
            "data_analytics": skill_reason(
                "Security investigations involve logs, alerts, access records, and other "
                "evidence. Analytical skill helps separate a real warning from normal noise.",
                "Analyze a synthetic event log, group related events, and write an "
                "evidence-based incident timeline.",
                ["onet_cyber"],
            ),
            "communication": skill_reason(
                "O*NET includes documenting security measures, discussing violations, "
                "training users, and promoting security awareness. Controls fail when "
                "people do not understand what they must do.",
                "Write a short incident summary for a manager and a separate set of clear "
                "actions for affected users.",
                ["onet_cyber"],
            ),
        },
        "ai_explanation": (
            "AI can help summarize alerts, group similar events, or draft a first incident "
            "note. Attackers can also use AI, and AI-generated security advice can be "
            "wrong. A competitive security analyst understands the system, verifies the "
            "evidence, protects sensitive data, and remains accountable for the response."
        ),
        "ai_actions": [
            "Practice checking AI suggestions against logs, approved procedures, and trusted guidance.",
            "Learn how privacy and security rules affect the data that may be sent to an AI tool.",
            "Strengthen incident communication so technical findings lead to timely action.",
        ],
        "role_sources": ["onet_cyber"],
        "ai_sources": ["wef_skills", "ilo_genai", "onet_cyber"],
    },
    "cloud_solutions_engineer": {
        "fit": (
            "This job is a close match when you can understand how applications, networks, "
            "data, and cloud services work together. The work is to design a dependable "
            "technical solution, test it, document it, and help other people use or "
            "maintain it."
        ),
        "typical_work": [
            "Translate user or business needs into a technical system design.",
            "Test system stability, security, compatibility, and scalability.",
            "Document the design and guide implementation or troubleshooting.",
        ],
        "skill_reasons": {
            "software_cloud": skill_reason(
                "Systems engineers design, integrate, configure, test, and troubleshoot "
                "computer systems. This is the technical foundation of the role.",
                "Design a small cloud solution, show how its components connect, and "
                "document how it is deployed and monitored.",
                ["onet_systems"],
            ),
            "cybersecurity_risk": skill_reason(
                "O*NET includes secure-system guidance, security analysis, and checking "
                "architecture for security. A working solution is not acceptable if it "
                "exposes data or access.",
                "Add identity, permissions, encryption, logging, and recovery controls to "
                "a sample cloud design.",
                ["onet_systems"],
            ),
            "project_change": skill_reason(
                "Solutions are implemented through coordinated changes, tests, schedules, "
                "and handoffs. Planning reduces outages and unclear ownership.",
                "Create a short migration plan with steps, risks, owners, tests, and a "
                "rollback procedure.",
                ["onet_systems"],
            ),
            "communication": skill_reason(
                "Systems engineers gather requirements, give technical guidance, and "
                "communicate designs through reports and presentations.",
                "Explain the same architecture twice: once for engineers and once for a "
                "non-technical decision-maker.",
                ["onet_systems"],
            ),
        },
        "ai_explanation": (
            "AI can draft configuration examples, documentation, or troubleshooting ideas. "
            "It cannot see every dependency, policy, cost, or failure mode in a real "
            "environment. A competitive engineer uses AI as a helper, then tests the "
            "solution and accepts responsibility for security and reliability."
        ),
        "ai_actions": [
            "Use AI-generated code only in a test environment before considering production use.",
            "Build stronger skills in architecture trade-offs, security, testing, and recovery.",
            "Document what was verified, what remains uncertain, and who approves the change.",
        ],
        "role_sources": ["onet_systems"],
        "ai_sources": ["wef_skills", "ilo_genai", "onet_systems"],
    },
    "project_manager": {
        "fit": (
            "This job is a close match when you can organize work, keep people aligned, "
            "and move a goal from idea to completion. Project managers do not have to be "
            "the deepest technical expert. They make the plan visible, manage risks and "
            "resources, and help the team make timely decisions."
        ),
        "typical_work": [
            "Build and update plans covering scope, schedule, people, costs, and deliverables.",
            "Coordinate stakeholders, resolve problems, and communicate progress.",
            "Track risks, quality, budget, and whether promised outcomes were delivered.",
        ],
        "skill_reasons": {
            "project_change": skill_reason(
                "O*NET describes planning schedules, staffing, procurement, budgets, "
                "milestones, and deliverables as central project-management work.",
                "Build a project plan with clear outcomes, owners, milestones, risks, and "
                "a method for controlling changes.",
                ["onet_project"],
            ),
            "communication": skill_reason(
                "Project managers gather requirements, facilitate meetings, negotiate "
                "resources, report status, and explain problems to different stakeholders.",
                "Prepare a one-page status update that clearly states progress, risks, "
                "decisions needed, and next actions.",
                ["onet_project"],
            ),
            "people_coaching": skill_reason(
                "The role assigns responsibilities, guides team members, gives feedback, "
                "and brings different people together to complete shared work.",
                "Run a short team review that identifies a blocker without blaming people "
                "and agrees on an owner and due date.",
                ["onet_project"],
            ),
            "finance_commercial": skill_reason(
                "Project managers monitor costs, prepare budget estimates, and make "
                "trade-offs when time, money, and resources are limited.",
                "Create a simple project budget and explain how one delay or scope change "
                "would affect cost and delivery.",
                ["onet_project"],
            ),
        },
        "ai_explanation": (
            "AI can draft minutes, summarize updates, or produce a first schedule. It "
            "cannot secure genuine agreement, understand every stakeholder concern, or "
            "take responsibility for a difficult trade-off. A competitive project manager "
            "uses AI to reduce administration and spends more time on judgment, alignment, "
            "risk, and delivery."
        ),
        "ai_actions": [
            "Automate low-risk administration while checking names, dates, owners, and decisions.",
            "Strengthen facilitation, negotiation, risk thinking, and change leadership.",
            "Keep a human-approved record of decisions rather than treating AI text as authority.",
        ],
        "role_sources": ["onet_project", "pmi_talent"],
        "ai_sources": ["wef_skills", "ilo_genai", "onet_project"],
    },
    "digital_marketing_strategist": {
        "fit": (
            "This job is a close match when you can understand an audience, create a clear "
            "message, and measure whether a campaign worked. Strong digital marketing "
            "combines customer research, creative communication, and evidence from data."
        ),
        "typical_work": [
            "Research customer needs, competitors, channels, and market conditions.",
            "Plan and produce campaign messages or digital experiences for a target audience.",
            "Measure results and adjust the campaign using evidence rather than opinion alone.",
        ],
        "skill_reasons": {
            "customer_research": skill_reason(
                "O*NET lists collecting information about customer needs, preferences, "
                "buying habits, and opinions as core marketing research work.",
                "Interview or survey a small target group and turn the findings into clear "
                "audience needs and campaign assumptions.",
                ["onet_market"],
            ),
            "communication": skill_reason(
                "Marketing specialists translate research into reports, proposals, and "
                "messages that influence a target audience and guide managers.",
                "Write one campaign message for a defined audience and explain the evidence "
                "behind the wording and channel choice.",
                ["onet_market"],
            ),
            "creative_design": skill_reason(
                "Campaigns need ideas and content that make the message easy to notice and "
                "understand. O*NET includes thinking creatively and knowledge of media and "
                "communication methods.",
                "Create a small campaign concept with a message, visual direction, channel, "
                "and reason each element suits the audience.",
                ["onet_market"],
            ),
            "data_analytics": skill_reason(
                "O*NET includes analyzing customer data, web metrics, trends, and campaign "
                "effectiveness. Measurement is needed to learn what actually worked.",
                "Define a campaign goal and dashboard, then compare results with a baseline "
                "instead of reporting clicks without context.",
                ["onet_market"],
            ),
        },
        "ai_explanation": (
            "AI can produce many draft messages and images quickly. It does not automatically "
            "know which claim is accurate, respectful, distinctive, or useful to a specific "
            "customer. A competitive marketer uses AI for options and speed, then applies "
            "customer evidence, brand judgment, legal checks, and performance measurement."
        ),
        "ai_actions": [
            "Use customer research to guide prompts and reject generic AI output.",
            "Check generated claims, sources, permissions, bias, and brand consistency.",
            "Run controlled tests and use results—not volume of generated content—to judge value.",
        ],
        "role_sources": ["onet_market"],
        "ai_sources": ["wef_skills", "ilo_genai", "onet_market"],
    },
    "people_analytics_specialist": {
        "fit": (
            "This job is a close match when you can work responsibly with workforce data "
            "and explain what it may mean for employees and managers. The role connects "
            "people knowledge with analysis; it should support better questions and fairer "
            "decisions, not reduce a person to a score."
        ),
        "typical_work": [
            "Prepare and analyze workforce measures such as hiring, retention, or development.",
            "Explain findings and limitations to HR leaders and managers.",
            "Protect employee information and check whether measures could create unfair conclusions.",
        ],
        "skill_reasons": {
            "data_analytics": skill_reason(
                "O*NET includes analyzing employment-related data and preparing reports. "
                "Analytical skill is needed to calculate measures correctly and avoid "
                "confusing correlation with cause.",
                "Build a workforce dashboard with documented definitions, missing-data "
                "checks, and a plain-language note about what cannot be concluded.",
                ["onet_hr", "shrm_people"],
            ),
            "people_coaching": skill_reason(
                "Workforce numbers only make sense with knowledge of hiring, performance, "
                "development, employee relations, and how policies affect people.",
                "Turn one workforce finding into a manager conversation and a supportive "
                "action rather than an automatic judgment about an employee.",
                ["onet_hr", "shrm_people"],
            ),
            "communication": skill_reason(
                "HR specialists explain policies, advise managers, provide training, and "
                "report employment information. Sensitive findings must be communicated "
                "clearly and carefully.",
                "Present a finding with its definition, evidence, uncertainty, possible "
                "human impact, and a responsible next step.",
                ["onet_hr"],
            ),
            "cybersecurity_risk": skill_reason(
                "People analytics uses sensitive employment information. Privacy, access "
                "control, appropriate use, and careful sharing are necessary to reduce harm.",
                "Design a simple data-access plan showing who may see individual records, "
                "who receives only grouped results, and when data should be deleted.",
                ["onet_hr", "ilo_genai"],
            ),
        },
        "ai_explanation": (
            "AI can help summarize survey comments or draft an analysis. It can also repeat "
            "bias, expose sensitive data, or create confident but unfair interpretations. "
            "A competitive people analyst combines data skill with privacy, fairness, "
            "workforce knowledge, and human review before any action affects employees."
        ),
        "ai_actions": [
            "Use anonymized or synthetic practice data and follow approved privacy controls.",
            "Test whether results differ unfairly across groups and investigate the cause.",
            "Keep consequential employee decisions with accountable humans and documented evidence.",
        ],
        "role_sources": ["onet_hr", "shrm_people"],
        "ai_sources": ["wef_skills", "ilo_genai", "onet_hr"],
    },
    "supply_chain_analyst": {
        "fit": (
            "This job is a close match when you can follow how materials or products move, "
            "measure what is going wrong, and improve the process. Supply-chain analysis "
            "connects data with practical operations such as inventory, purchasing, "
            "transport, delivery, and supplier performance."
        ),
        "typical_work": [
            "Track inventory, delivery, cost, capacity, and supplier performance.",
            "Find delays, shortages, waste, or process risks and investigate their causes.",
            "Recommend and coordinate an operational improvement with affected teams.",
        ],
        "skill_reasons": {
            "operations_quality": skill_reason(
                "O*NET describes coordinating the product life cycle, reviewing logistics "
                "performance, allocating materials, and improving the movement of goods.",
                "Map a simple order-to-delivery process, identify one failure point, and "
                "propose a measurable control or improvement.",
                ["onet_logistics"],
            ),
            "data_analytics": skill_reason(
                "Logistics work includes compiling and analyzing technical and performance "
                "data. Analysts need evidence to separate a recurring problem from a one-time event.",
                "Analyze synthetic inventory or delivery data and show the main cause of "
                "late or incomplete orders.",
                ["onet_logistics"],
            ),
            "project_change": skill_reason(
                "Supply-chain improvements cross departments and suppliers. O*NET includes "
                "plans, schedules, responsibility matrices, and coordination activities.",
                "Create a small improvement plan with owners, milestones, risks, and a "
                "before-and-after measure.",
                ["onet_logistics"],
            ),
            "communication": skill_reason(
                "Logisticians explain solutions to customers and management and coordinate "
                "with other departments, suppliers, and service providers.",
                "Explain a shortage or delay using evidence, business impact, available "
                "choices, and the decision needed.",
                ["onet_logistics"],
            ),
        },
        "ai_explanation": (
            "AI can help forecast demand, classify issues, or compare scenarios. A forecast "
            "can still fail when data is poor or when a supplier, weather event, policy, or "
            "local constraint changes. A competitive analyst checks assumptions, understands "
            "the real process, and prepares people to act when the model is wrong."
        ),
        "ai_actions": [
            "Learn to compare an AI or statistical forecast with a simple baseline.",
            "Document assumptions and create an exception plan for shortages or disruptions.",
            "Keep supplier, cost, customer, and operational data within approved systems.",
        ],
        "role_sources": ["onet_logistics", "ascm_cpim"],
        "ai_sources": ["wef_skills", "ilo_genai", "onet_logistics"],
    },
    "fpa_analyst": {
        "fit": (
            "This job is a close match when you understand money, can analyze business "
            "performance, and can explain the financial effect of a decision. FP&A work "
            "connects budgets and forecasts with the operational story behind the numbers."
        ),
        "typical_work": [
            "Prepare budgets, forecasts, scenarios, and performance comparisons.",
            "Investigate why actual results differ from the plan.",
            "Explain financial risks, trade-offs, and choices to managers.",
        ],
        "skill_reasons": {
            "finance_commercial": skill_reason(
                "Financial analysts use financial models, assess performance and risk, and "
                "prepare plans. Finance knowledge is needed to use the right definitions "
                "and understand the business effect of a number.",
                "Build a simple forecast with revenue, cost, cash, and clearly stated assumptions.",
                ["onet_finance"],
            ),
            "data_analytics": skill_reason(
                "The work requires quantitative analysis of financial and operational data, "
                "including trends, comparisons, and model-based scenarios.",
                "Compare actual results with budget, identify the largest variance, and "
                "trace it to an operational driver.",
                ["onet_finance"],
            ),
            "communication": skill_reason(
                "O*NET includes client presentations and oral or written reports. Leaders "
                "need the meaning and decision, not only a spreadsheet.",
                "Present a forecast in plain language: what changed, why it matters, what is "
                "uncertain, and what decision is requested.",
                ["onet_finance"],
            ),
            "ai_automation": skill_reason(
                "AI and automation can speed up repetitive reporting and scenario drafts, "
                "but financial outputs can cause harm if assumptions or calculations are wrong.",
                "Automate one repeatable report and add reconciliation, reasonableness, and "
                "approval checks before it is used.",
                ["wef_skills", "ilo_genai", "onet_finance"],
            ),
        },
        "ai_explanation": (
            "AI can draft commentary, formulas, or scenarios. It cannot approve an assumption, "
            "own a budget, or understand every business constraint. A competitive FP&A "
            "professional uses AI to reduce repetitive work, then checks the numbers and "
            "focuses on business judgment, scenarios, and clear advice."
        ),
        "ai_actions": [
            "Reconcile AI-assisted work to approved source data and financial definitions.",
            "Make every important assumption visible and test more than one scenario.",
            "Keep financial and commercial information inside approved tools and review paths.",
        ],
        "role_sources": ["onet_finance"],
        "ai_sources": ["wef_skills", "ilo_genai", "onet_finance"],
    },
    "laboratory_quality_specialist": {
        "fit": (
            "This job is a close match when you can perform scientific work carefully and "
            "follow quality procedures. The role connects hands-on testing with accurate "
            "records, equipment control, investigation of unusual results, and evidence "
            "that a laboratory process is dependable."
        ),
        "typical_work": [
            "Test samples using approved methods and compare results with specifications.",
            "Maintain equipment, records, safety, cleanliness, and controlled procedures.",
            "Investigate unusual results and recommend corrective or preventive action.",
        ],
        "skill_reasons": {
            "scientific_laboratory": skill_reason(
                "Quality-control analysts conduct laboratory tests, interpret results, "
                "maintain equipment, and validate methods. Scientific knowledge is needed "
                "to perform the method and recognize when a result is not credible.",
                "Run a documented test on a safe practice sample, record observations, and "
                "explain the scientific basis of the method.",
                ["onet_quality", "prc_chem_lab"],
            ),
            "operations_quality": skill_reason(
                "The job compares results with specifications, monitors procedures, writes "
                "standard procedures, supports audits, and investigates failures. ISO/IEC "
                "17025 also focuses on competent and consistent laboratory operation.",
                "Create a simple deviation investigation showing the problem, evidence, "
                "likely cause, corrective action, and effectiveness check.",
                ["onet_quality", "iso_17025"],
            ),
            "data_analytics": skill_reason(
                "Laboratory quality work compiles test data, analyzes trends, checks control "
                "limits, and investigates out-of-specification or questionable results.",
                "Plot a synthetic set of quality-control results and explain whether a "
                "change appears meaningful and what should be checked next.",
                ["onet_quality"],
            ),
            "communication": skill_reason(
                "O*NET lists technical reports, deviation reports, protocols, procedures, "
                "training, and liaison work. Clear records allow another qualified person "
                "to understand what happened and repeat or review the work.",
                "Write a short laboratory report with method, result, limitation, deviation, "
                "and next action in clear language.",
                ["onet_quality", "iso_17025"],
            ),
        },
        "ai_explanation": (
            "AI can help organize literature, draft a procedure outline, or look for patterns "
            "in non-sensitive data. It cannot physically handle the sample, observe the "
            "instrument, guarantee traceability, or sign professional and regulatory "
            "responsibility. A competitive laboratory professional combines digital tools "
            "with hands-on competence, quality judgment, safety, and complete records."
        ),
        "ai_actions": [
            "Use AI only with approved, non-sensitive data and verify every technical statement.",
            "Strengthen method validation, data integrity, investigations, and equipment knowledge.",
            "Keep final scientific conclusions and regulated approvals with qualified humans.",
        ],
        "role_sources": ["onet_quality", "iso_17025", "prc_chem_lab"],
        "ai_sources": ["wef_skills", "ilo_genai", "onet_quality", "iso_17025"],
    },
    "customer_experience_manager": {
        "fit": (
            "This job is a close match when you can listen to customers, explain problems "
            "clearly, and improve the service that teams deliver. Customer-experience work "
            "looks across the whole journey, not only one complaint, and turns repeated "
            "customer evidence into operational change."
        ),
        "typical_work": [
            "Collect and interpret customer feedback, behavior, and service measures.",
            "Find recurring problems across channels or stages of the customer journey.",
            "Coordinate teams to improve the service and measure whether the change helped.",
        ],
        "skill_reasons": {
            "customer_research": skill_reason(
                "Market-research work includes collecting customer needs, opinions, "
                "preferences, and satisfaction data. CX decisions need evidence from the "
                "people who actually use the service.",
                "Combine a small set of interviews, complaints, and service measures into "
                "a journey map with evidence for each pain point.",
                ["onet_market"],
            ),
            "communication": skill_reason(
                "CX managers must listen carefully, explain customer impact, and make a "
                "clear case for action across teams with different priorities.",
                "Present one customer problem using a real example, supporting data, business "
                "impact, and a specific decision request.",
                ["onet_market", "onet_operations"],
            ),
            "people_coaching": skill_reason(
                "Service improvement depends on frontline and support teams changing how "
                "work is done. Operations management includes guiding, developing, and "
                "coordinating people.",
                "Coach a team through one service breakdown, focusing on the process and "
                "the next behavior rather than blame.",
                ["onet_operations"],
            ),
            "operations_quality": skill_reason(
                "A customer problem often comes from a process, handoff, control, or capacity "
                "issue. Operational skill is needed to make an improvement reliable, not temporary.",
                "Map the service process, identify the failure point, change one control, "
                "and track a customer and an operational measure.",
                ["onet_operations", "onet_market"],
            ),
        },
        "ai_explanation": (
            "AI can summarize feedback, suggest reply drafts, or identify repeated themes. "
            "It may miss context, emotion, accessibility needs, or the seriousness of an "
            "individual case. A competitive CX professional uses AI for scale while keeping "
            "human listening, fair judgment, escalation, and service accountability."
        ),
        "ai_actions": [
            "Check AI-generated themes against original feedback and multiple customer groups.",
            "Protect customer information and avoid placing private conversations in unapproved tools.",
            "Measure whether an AI-enabled service change improves outcomes without excluding people.",
        ],
        "role_sources": ["onet_market", "onet_operations"],
        "ai_sources": ["wef_skills", "ilo_genai", "onet_market", "onet_operations"],
    },
    "ux_researcher": {
        "fit": (
            "This job is a close match when you are curious about how people use a product "
            "and can turn observation into design guidance. UX researchers plan studies, "
            "listen without leading the participant, analyze evidence, and explain what the "
            "product team should learn or test next."
        ),
        "typical_work": [
            "Choose a suitable research method and recruit relevant participants.",
            "Conduct interviews or usability studies and analyze recurring evidence.",
            "Explain user needs, uncertainty, and design implications to a product team.",
        ],
        "skill_reasons": {
            "customer_research": skill_reason(
                "Market-research tasks include collecting needs and opinions and designing "
                "surveys or other data-collection methods. UX research applies those skills "
                "to product and service use.",
                "Plan and conduct five usability sessions on a safe prototype, then separate "
                "observed behavior from your interpretation.",
                ["onet_market", "onet_digital_design"],
            ),
            "communication": skill_reason(
                "Researchers interview participants and translate complex findings into "
                "clear written or visual guidance for designers and decision-makers.",
                "Write a research readout with the question, method, evidence, limitation, "
                "and two prioritized recommendations.",
                ["onet_market", "onet_digital_design"],
            ),
            "creative_design": skill_reason(
                "Researchers work with prototypes, user flows, interfaces, and design "
                "choices. Design literacy helps them frame realistic tasks and make findings actionable.",
                "Create or use a simple prototype and explain how each research finding "
                "could affect the design without pretending the finding is a final answer.",
                ["onet_digital_design"],
            ),
            "data_analytics": skill_reason(
                "Research requires organizing observations and, when appropriate, survey "
                "or product data. Analytical skill helps identify patterns without "
                "overstating what a small study proves.",
                "Code interview notes or usability problems, count recurring patterns, and "
                "state the sample and limitations beside the result.",
                ["onet_market"],
            ),
        },
        "ai_explanation": (
            "AI can transcribe, organize notes, or suggest possible themes. It may erase "
            "nuance, invent a pattern, or reproduce bias. A competitive UX researcher uses "
            "AI to reduce administration while preserving participant context, consent, "
            "privacy, careful interpretation, and direct observation."
        ),
        "ai_actions": [
            "Check every AI-generated theme against the original research evidence.",
            "Remove or protect participant identifiers before using approved analysis tools.",
            "Strengthen interviewing, study design, accessibility, and research ethics.",
        ],
        "role_sources": ["onet_market", "onet_digital_design"],
        "ai_sources": ["wef_skills", "ilo_genai", "onet_market", "onet_digital_design"],
    },
}


LEARNING_OPTIONS: dict[str, list[dict[str, str]]] = {
    "data_bi_analyst": [
        {
            "type": "Certification",
            "name": "Microsoft Certified: Power BI Data Analyst Associate",
            "provider": "Microsoft",
            "url": "https://learn.microsoft.com/en-us/credentials/certifications/data-analyst-associate/",
            "fit": "Validates preparing, modeling, visualizing, analyzing, and securing data in Power BI.",
            "eligibility": "Intermediate credential; Microsoft lists no formal work-experience prerequisite. Review the current PL-300 study guide.",
        },
        {
            "type": "Course",
            "name": "PL-300: Design and manage analytics solutions using Power BI",
            "provider": "Microsoft Learn",
            "url": "https://learn.microsoft.com/en-us/training/courses/pl-300t00",
            "fit": "Official preparation covering the Power BI workflow assessed by PL-300.",
            "eligibility": "Review the course prerequisites and current delivery options on Microsoft Learn.",
        },
    ],
    "cybersecurity_analyst": [
        {
            "type": "Certification",
            "name": "Certified in Cybersecurity (CC)",
            "provider": "ISC2",
            "url": "https://www.isc2.org/certifications/cc",
            "fit": "Entry-level coverage of security principles, incident response, access controls, networks, and security operations.",
            "eligibility": "ISC2 states that no work experience is required. Confirm current exam, training, and membership terms.",
        },
        {
            "type": "Official training",
            "name": "ISC2 CC Online Self-Paced Training",
            "provider": "ISC2",
            "url": "https://www.isc2.org/certifications/cc",
            "fit": "Official training aligned with the current CC exam domains.",
            "eligibility": "Availability and pricing can change; verify them on the ISC2 page before enrolling.",
        },
    ],
    "cloud_solutions_engineer": [
        {
            "type": "Certification",
            "name": "AWS Certified Solutions Architect – Associate",
            "provider": "Amazon Web Services",
            "url": "https://aws.amazon.com/certification/certified-solutions-architect-associate/",
            "fit": "Covers secure, resilient, high-performing, and cost-optimized cloud solution design.",
            "eligibility": "No formal prerequisite; AWS recommends about one year of hands-on cloud solution-design experience.",
        },
        {
            "type": "Learning path",
            "name": "Microsoft Azure Fundamentals: Describe cloud concepts",
            "provider": "Microsoft Learn",
            "url": "https://learn.microsoft.com/en-us/training/paths/microsoft-azure-fundamentals-describe-cloud-concepts/",
            "fit": "A beginner path for cloud concepts before deeper platform architecture study.",
            "eligibility": "Beginner learning path; check the live Microsoft Learn page for current modules.",
        },
    ],
    "project_manager": [
        {
            "type": "Certification",
            "name": "Certified Associate in Project Management (CAPM)®",
            "provider": "Project Management Institute",
            "url": "https://www.pmi.org/certifications/certified-associate-capm",
            "fit": "Builds foundational knowledge in predictive, agile, and business-analysis ways of working.",
            "eligibility": "PMI requires a secondary degree or equivalent plus 23 hours of project-management education; no work experience is required.",
        },
        {
            "type": "Advanced certification",
            "name": "Project Management Professional (PMP)®",
            "provider": "Project Management Institute",
            "url": "https://www.pmi.org/certifications/project-management-pmp",
            "fit": "For professionals who can document substantial responsibility for leading projects.",
            "eligibility": "Experience and training requirements vary by education. Verify the current PMI eligibility route before applying.",
        },
    ],
    "digital_marketing_strategist": [
        {
            "type": "Professional certificate",
            "name": "Google Digital Marketing & E-commerce Certificate",
            "provider": "Google",
            "url": "https://grow.google/certificates/digital-marketing-ecommerce/",
            "fit": "Covers campaigns, customer engagement, analytics, e-commerce, and practical AI use in marketing.",
            "eligibility": "Google describes it as foundational and requiring no previous experience.",
        },
        {
            "type": "Certification",
            "name": "Meta Certified Digital Marketing Associate",
            "provider": "Meta Blueprint",
            "url": "https://www.facebookblueprint.com/student/path/517001-get-certified-as-digital-marketing-associate",
            "fit": "Covers Meta advertising fundamentals, targeting, creative, optimization, and measurement.",
            "eligibility": "Entry-level credential; confirm current exam language, delivery, and regional availability.",
        },
    ],
    "people_analytics_specialist": [
        {
            "type": "Specialty credential",
            "name": "People Analytics Specialty Credential",
            "provider": "Society for Human Resource Management",
            "url": "https://www.shrm.org/credentials/specialty-credentials/people-analytics-credential",
            "fit": "Combines data literacy, workforce metrics, applied analysis, and communication of people insights.",
            "eligibility": "SHRM states that SHRM-CP or SHRM-SCP certification is not required; verify current package components and availability.",
        },
        {
            "type": "Course",
            "name": "Transform business workflows with generative AI",
            "provider": "Microsoft Learn",
            "url": "https://learn.microsoft.com/en-us/training/courses/ab-730t00",
            "fit": "Builds practical, no-code AI workflow skills applicable to HR reporting and decision support.",
            "eligibility": "Beginner course for business users; review current product-access requirements.",
        },
    ],
    "supply_chain_analyst": [
        {
            "type": "Certification",
            "name": "APICS Certified in Planning and Inventory Management (CPIM)",
            "provider": "Association for Supply Chain Management",
            "url": "https://www.ascm.org/learning-development/certifications-credentials/cpim/",
            "fit": "Covers strategy alignment, S&OP, demand, supply, inventory, quality improvement, and technology.",
            "eligibility": "Review the current CPIM exam version, bundle, testing, and maintenance requirements with ASCM.",
        },
        {
            "type": "Course",
            "name": "Transform business workflows with generative AI",
            "provider": "Microsoft Learn",
            "url": "https://learn.microsoft.com/en-us/training/courses/ab-730t00",
            "fit": "Supports responsible automation of recurring analysis, reporting, and workflow tasks.",
            "eligibility": "Beginner course for business users; confirm current access and delivery options.",
        },
    ],
    "fpa_analyst": [
        {
            "type": "Certification",
            "name": "Certified Management Accountant (CMA)®",
            "provider": "Institute of Management Accountants",
            "url": "https://www.imanet.org/ima-certifications/cma-certification",
            "fit": "Covers planning, performance, analytics, controls, financial decision-making, and strategy.",
            "eligibility": "IMA requires membership, qualifying education, two years of relevant experience, and both exam parts.",
        },
        {
            "type": "Certification",
            "name": "Certified Corporate FP&A Professional (FPAC)",
            "provider": "Association for Financial Professionals",
            "url": "https://fpacert.afponline.org/about-exam/eligibility",
            "fit": "Focuses on forecasting, modeling, planning, analytics, and business partnership.",
            "eligibility": "AFP applies education, relevant full-time experience, ethics, and two-part examination requirements.",
        },
    ],
    "laboratory_quality_specialist": [
        {
            "type": "Certification",
            "name": "Certified Quality Improvement Associate (CQIA)",
            "provider": "American Society for Quality",
            "url": "https://www.asq.org/cert/quality-improvement-associate",
            "fit": "Builds foundational quality tools, team participation, and improvement-method knowledge.",
            "eligibility": "ASQ currently requires two years of full-time paid experience or qualifying higher education. Verify the current handbook.",
        },
        {
            "type": "Course",
            "name": "Quality 101: CQIA Certification Preparation",
            "provider": "American Society for Quality",
            "url": "https://asq.org/training/quality-101--certified-quality-improvement-associate-certification-preparation-spcqia2020asq",
            "fit": "Official introductory quality course aligned with the CQIA body of knowledge.",
            "eligibility": "ASQ lists no prerequisite; course completion does not guarantee exam success.",
        },
    ],
    "customer_experience_manager": [
        {
            "type": "Certification",
            "name": "Certified Customer Experience Professional (CCXP)",
            "provider": "Customer Experience Professionals Association",
            "url": "https://cxpaglobal.org/get-certified/get-started",
            "fit": "Validates experience across customer strategy, culture, insights, design, and measurement.",
            "eligibility": "CXPA lists education-and-experience routes; this is intended for experienced CX practitioners.",
        },
        {
            "type": "Course directory",
            "name": "CXPA Recognized Training Providers",
            "provider": "Customer Experience Professionals Association",
            "url": "https://cxpaglobal.org/get-certified/recognized-training-providers",
            "fit": "Lists independently reviewed providers whose training aligns with the CXPA framework.",
            "eligibility": "Training is supplementary and does not replace CCXP experience or examination requirements.",
        },
    ],
    "ux_researcher": [
        {
            "type": "Professional certificate",
            "name": "Google UX Design Certificate",
            "provider": "Google",
            "url": "https://grow.google/certificates/ux-design/",
            "fit": "Covers user research, accessibility, wireframes, prototypes, usability testing, and portfolio work.",
            "eligibility": "Google describes it as foundational and requiring no previous experience.",
        },
        {
            "type": "Certification",
            "name": "UX Certification",
            "provider": "Nielsen Norman Group",
            "url": "https://www.nngroup.com/ux-certification/",
            "fit": "Course- and exam-based professional development across UX research and design topics.",
            "eligibility": "Review the current course-count, examination, pricing, and delivery requirements directly with NN/g.",
        },
    ],
}


UNIVERSAL_AI_COURSE = {
    "type": "AI course",
    "name": "Transform business workflows with generative AI",
    "provider": "Microsoft Learn",
    "url": "https://learn.microsoft.com/en-us/training/courses/ab-730t00",
    "fit": "A beginner, no-code course on applying generative AI to workflows, decisions, and business outcomes.",
    "eligibility": "Designed for business users across functions; confirm current product-access and delivery requirements.",
}


LEGACY_LEADERSHIP_PATHWAYS: dict[str, dict[str, Any]] = {
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
    "laboratory_quality_specialist": {
        "title": "Laboratory & Scientific Operations Leadership Pathway",
        "summary": (
            "Build from chemistry or laboratory practice toward supervision of "
            "scientific work, laboratory quality, safety, compliance, and defensible "
            "technical decisions."
        ),
        "progression": (
            "Senior Chemist or Laboratory Analyst → Laboratory Supervisor or Quality "
            "Lead → Laboratory Manager or Head of Chemical Laboratory"
        ),
        "focus": (
            "Strengthen method and quality-system ownership, laboratory safety, "
            "technical review, coaching, workload decisions, ethical escalation, and "
            "accountability under Philippine chemistry requirements."
        ),
        "proof": (
            "Lead a bounded laboratory-quality or safety improvement using sanitized "
            "information. Document the problem, applicable procedure or standard, "
            "contributors, risk decision, controlled change, verification result, and "
            "management follow-through."
        ),
        "core_skills": ["scientific_laboratory"],
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
    """Generate balanced, overlapping profiles using only skills and experience."""

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    profile_number = 1
    career_ids = list(JOB_PROTOTYPES)

    for career_index, (career_id, prototype) in enumerate(JOB_PROTOTYPES.items()):
        adjacent = JOB_PROTOTYPES[
            career_ids[(career_index + 1) % len(career_ids)]
        ]
        for _ in range(per_career):
            # Some profiles blend with a neighboring job family to create
            # realistic ambiguity instead of perfectly separated classes.
            blend = (
                float(rng.uniform(0.08, 0.30))
                if rng.random() < 0.32
                else 0.0
            )
            experience_center = (
                prototype["experience"] * (1 - blend)
                + adjacent["experience"] * blend
            )
            years = float(np.clip(rng.normal(experience_center, 2.4), 0, 30))
            row: dict[str, Any] = {
                "profile_id": f"SKILL-{profile_number:04d}",
                "years_experience": round(years, 1),
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
                no_experience_probability = 0.02 + 0.18 * (1 - mixed_center) ** 2
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
    expected_rows = len(JOB_PROTOTYPES) * per_career
    if (
        len(frame) != expected_rows
        or not frame["recommended_career"].value_counts().eq(per_career).all()
    ):
        raise RuntimeError("Synthetic data integrity check failed.")
    return frame


@dataclass
class CareerEngine:
    model: Pipeline
    synthetic_profiles: pd.DataFrame

    @classmethod
    def train(cls) -> "CareerEngine":
        profiles = generate_synthetic_profiles()
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=500,
                        class_weight="balanced",
                        min_samples_leaf=2,
                        random_state=SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        model.fit(profiles[NUMERIC_FEATURES], profiles["recommended_career"])
        return cls(model=model, synthetic_profiles=profiles)

    def recommend(
        self,
        profile: dict[str, Any],
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        validate_profile(profile)
        input_frame = pd.DataFrame([profile])[NUMERIC_FEATURES]
        probabilities = self.model.predict_proba(input_frame)[0]
        class_ids = self.model.named_steps["model"].classes_
        results: list[dict[str, Any]] = []

        for career_id, model_probability in zip(class_ids, probabilities):
            career_id = str(career_id)
            career = CAREERS[career_id]
            prototype = JOB_PROTOTYPES[career_id]
            role_guidance = ROLE_GUIDANCE[career_id]
            current = career["current_demand"]
            future = career["future_demand"]
            skill_alignment = skill_alignment_for(profile, career_id)
            core_skill_coverage = core_skill_coverage_for(profile, career_id)
            experience_fit = float(
                np.exp(
                    -abs(
                        float(profile["years_experience"])
                        - float(prototype["experience"])
                    )
                    / max(3.0, float(prototype["experience"]) * 0.80)
                )
            )
            ai_target = max(
                0.20,
                float(
                    prototype["skills"][
                        list(SKILLS).index("ai_automation")
                    ]
                ),
            )
            ai_skill_coverage = min(
                1.0,
                (float(profile["ai_automation"]) / 5) / ai_target,
            )
            ai_competitiveness = (
                0.45 * float(future["score"])
                + 0.30 * float(current["score"])
                + 0.25 * ai_skill_coverage
            )
            score = 100 * (
                0.50 * skill_alignment
                + 0.15 * core_skill_coverage
                + 0.15 * float(model_probability)
                + 0.08 * experience_fit
                + 0.05 * float(current["score"])
                + 0.07 * float(future["score"])
            )
            learning_options = list(LEARNING_OPTIONS[career_id])
            if not any(
                option["url"] == UNIVERSAL_AI_COURSE["url"]
                for option in learning_options
            ):
                learning_options.append(dict(UNIVERSAL_AI_COURSE))

            results.append(
                {
                    "career_id": career_id,
                    "career": career["title"],
                    "summary": career["summary"],
                    "display_career": career["title"],
                    "display_summary": career["summary"],
                    "recommendation_score": round(score, 1),
                    "synthetic_model_fit": round(
                        100 * float(model_probability),
                        1,
                    ),
                    "skill_alignment": round(100 * skill_alignment, 1),
                    "core_skill_coverage": round(
                        100 * core_skill_coverage,
                        1,
                    ),
                    "experience_fit": round(100 * experience_fit, 1),
                    "ai_competitiveness": round(
                        100 * ai_competitiveness,
                        1,
                    ),
                    "experience_guidance": experience_guidance_for(
                        float(profile["years_experience"])
                    ),
                    "matched_skills": matched_skills_for(profile, career_id),
                    "skill_gaps": skill_gaps_for(profile, career_id, limit=4),
                    "application_contexts": prototype["application_contexts"],
                    "role_guidance": role_guidance,
                    "learning_options": learning_options,
                    **career,
                }
            )

        ranked = sorted(
            results,
            key=lambda item: (
                item["recommendation_score"],
                item["synthetic_model_fit"],
            ),
            reverse=True,
        )
        supported = [
            item
            for item in ranked
            if item["skill_alignment"] >= 40
            and item["core_skill_coverage"] >= 25
            and item["recommendation_score"] >= 55
        ]
        if not supported:
            raise ValueError(
                "The app could not find a sufficiently close job match in its "
                "current catalog. It has abstained instead of forcing an unrelated "
                "recommendation. Add only skills you can demonstrate, or treat your "
                "target occupation as outside the present catalog."
            )
        return supported[:top_k]


def validate_profile(profile: dict[str, Any]) -> None:
    expected = NUMERIC_FEATURES
    missing = [field for field in expected if field not in profile]
    if missing:
        raise ValueError(f"Missing profile fields: {', '.join(missing)}")
    years = float(profile["years_experience"])
    if not 0 <= years <= 50:
        raise ValueError("Years of experience must be between 0 and 50.")
    for skill in SKILLS:
        if not 0 <= float(profile[skill]) <= 5:
            raise ValueError(f"{SKILLS[skill]['label']} must be between 0 and 5.")
    if not any(float(profile[skill]) > 0 for skill in SKILLS):
        raise ValueError(
            "Rate at least one skill above 0 so the app has evidence to match."
        )


def target_ratings(career_id: str) -> dict[str, int]:
    return {
        skill: int(np.clip(np.rint(5 * center), 1, 5))
        for skill, center in zip(SKILLS, JOB_PROTOTYPES[career_id]["skills"])
    }


def skill_alignment_for(profile: dict[str, Any], career_id: str) -> float:
    user = np.array([float(profile[skill]) / 5 for skill in SKILLS])
    target = np.array(JOB_PROTOTYPES[career_id]["skills"], dtype=float)
    if float(np.linalg.norm(user)) == 0:
        return 0.0
    cosine = float(
        np.dot(user, target)
        / (float(np.linalg.norm(user)) * float(np.linalg.norm(target)))
    )
    coverage = float(np.minimum(user, target).sum() / target.sum())
    return float(np.clip(0.65 * cosine + 0.35 * coverage, 0, 1))


def core_skill_coverage_for(profile: dict[str, Any], career_id: str) -> float:
    prototype = JOB_PROTOTYPES[career_id]
    target_by_skill = dict(zip(SKILLS, prototype["skills"]))
    coverage = [
        min(
            1.0,
            (float(profile[skill]) / 5)
            / max(0.20, float(target_by_skill[skill])),
        )
        for skill in prototype["core_skills"]
    ]
    return float(np.mean(coverage))


def matched_skills_for(
    profile: dict[str, Any],
    career_id: str,
    limit: int = 4,
) -> list[dict[str, Any]]:
    targets = target_ratings(career_id)
    skill_reasons = ROLE_GUIDANCE[career_id]["skill_reasons"]
    matches = [
        {
            "skill": skill,
            "label": SKILLS[skill]["label"],
            "current": float(profile[skill]),
            "target": target,
            "matched": min(float(profile[skill]), float(target)),
            **skill_reasons[skill],
        }
        for skill in skill_reasons
        if (target := targets[skill])
        if float(profile[skill]) > 0
    ]
    return sorted(
        matches,
        key=lambda item: (item["matched"], item["current"]),
        reverse=True,
    )[:limit]


def experience_guidance_for(years: float) -> str:
    if years < 2:
        return (
            "Investigate internships, apprenticeships, associate, coordinator, "
            "or junior versions of this job family."
        )
    if years < 5:
        return (
            "Investigate analyst or specialist roles and validate the required "
            "domain experience in current job descriptions."
        )
    if years < 8:
        return (
            "Investigate experienced analyst, senior specialist, or workstream-lead "
            "roles; title seniority still depends on demonstrated scope."
        )
    return (
        "Investigate senior specialist, lead, consulting, or management-adjacent "
        "roles, while validating domain depth and leadership requirements."
    )


def skill_gaps_for(
    profile: dict[str, Any],
    career_id: str,
    limit: int = 4,
) -> list[dict[str, Any]]:
    targets = target_ratings(career_id)
    skill_reasons = ROLE_GUIDANCE[career_id]["skill_reasons"]
    gaps = [
        {
            "skill": skill,
            "label": SKILLS[skill]["label"],
            "current": float(profile[skill]),
            "target": target,
            "gap": target - float(profile[skill]),
            **skill_reasons[skill],
        }
        for skill in skill_reasons
        if (target := targets[skill])
        if target - float(profile[skill]) > 0
    ]
    return sorted(gaps, key=lambda item: item["gap"], reverse=True)[:limit]
