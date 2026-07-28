# Hakbang PH

Hakbang PH is a skills-first career-recommendation research prototype. It ranks
cross-industry job families from only:

- total years of work experience; and
- 12 demonstrated-skill ratings from 0 (no experience) to 5 (independent,
  explainable performance).

Industry, current job title, employer, career goal, demographics, and protected
characteristics are excluded from training and recommendation scoring.

## Use the Google Colab notebook

The self-contained notebook is
[Hakbang_PH_Skills_First_Career_Recommender.ipynb](notebooks/Hakbang_PH_Skills_First_Career_Recommender.ipynb).
It generates 2,200 balanced synthetic profiles, benchmarks four classifiers,
fits the selected model, and returns up to three supported job-family matches
with research evidence, AI-era guidance, skill gaps, official certifications,
courses, and caveated credential-holder accounts.

After publishing this repository to GitHub, open:

```text
https://colab.research.google.com/github/OWNER/REPOSITORY/blob/main/notebooks/Hakbang_PH_Skills_First_Career_Recommender.ipynb
```

Replace `OWNER` and `REPOSITORY` with the GitHub owner and repository name.

Regenerate and validate the notebook:

```bash
python3 scripts/build_colab_notebook.py
python3 scripts/validate_colab_notebook.py
```

## Run the Streamlit app

```bash
python3 -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

The Streamlit entrypoint registers three separate pages:

- `/` — Career scan and recommendation results;
- `/method` — the current model, ranking, benchmark, and limitations; and
- `/evidence` — the dated research and official-source registry.

The Method page reads the weights, thresholds, model metrics, feature counts,
and dataset version from the same Python constants used by the recommender. The
Evidence page renders the same `SOURCES` registry referenced by recommendation
explanations, so neither page maintains a separate copy that can silently drift.

## Model and evidence controls

- The dataset has 200 synthetic profiles for each of 11 job families.
- Extra Trees, Random Forest, multinomial logistic regression, and
  distance-weighted 7-nearest-neighbors are compared with seeded stratified
  five-fold macro-F1 and accuracy.
- The Streamlit model uses Extra Trees, the strongest synthetic-label classifier
  in the current benchmark.
- Final ranking prioritizes skill alignment (50%) and core-skill coverage (15%);
  the synthetic model contributes 15%, experience 8%, and research-graded demand
  12%.
- Weak matches are withheld. The app may return fewer than three jobs.
- Career, demand, AI, credential, course, and practitioner statements are fixed
  and source-linked; no factual claim is generated at runtime.

Synthetic validation cannot establish real-world hiring or career-outcome
accuracy. Production use requires consented outcome data, current licensed
vacancy data, external and temporal validation, calibration, fairness testing,
accessibility review, and human oversight.

See [the model card](docs/MODEL_CARD.md) and
[the evidence ledger](docs/EVIDENCE_LEDGER.md).
