---
description: "Draft a 1-2 page individual reflection report for a team member (NUS-ISS Practice Module submission)"
agent: "agent"
argument-hint: "Team member name and their specific contributions to the project"
---
Draft a 1–2 page individual project reflection report, required as an additional submission for the NUS-ISS Practice Module, for the team member described in the user's request.

Ask the user for their name and their specific contributions if not already provided, then write the reflection to `docs/reflections/<name>_reflection.md` (create folders as needed).

Cover exactly these three points (per the assessment brief), grounded in this project's actual work — reference specific scripts/results from [README.md](../../README.md) rather than generic statements:

1. **Personal contribution to the group project** — which parts of the pipeline (manifest generation, baseline classifier, PatchCore detector, category classifier, ensemble, Streamlit app, a specific category's config/tuning, README findings, etc.) this person worked on, with concrete specifics.
2. **What was learnt that is most useful** — a genuine technical takeaway tied to a real finding from this project (e.g. transfer learning vs. from-scratch CNNs, why unsupervised PatchCore is more trustworthy than the leaky supervised baseline, per-category config decisions like `use_foreground_mask`) — pull from [README.md § Notes & Lessons Learned](../../README.md#notes--lessons-learned) if relevant to their contribution.
3. **How the knowledge/skills can be applied elsewhere** — a plausible, specific application to other situations or workplaces (not generic platitudes).

Keep the tone first-person and reflective, not a copy of the technical README — this is a personal account, not a system report.
