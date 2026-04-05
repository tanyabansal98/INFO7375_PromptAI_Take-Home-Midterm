# Chapter 19: Building Production Prompt Systems — From Prototype to Pipeline

**Book:** Design of Agentic Systems with Case Studies — Prompt Engineering with LLMs  
**Course:** INFO 7375 — Prompt Engineering for Generative AI, Northeastern University  
**Author:** Tanya Bansal  
**Chapter Type:** Type E — Operational / Production Practice

---

## Overview

This repository contains all deliverables for Chapter 19 of *Prompt Engineering with LLMs*. The chapter demonstrates that a prompt working in a prototype will fail silently in production without versioning, evaluation, fallback, and observability.

The chapter introduces a **Five Walls** architecture — Prompt Registry, Schema Gate, Evaluation Loop, Fallback Cascade, and Observability Layer — each blocking a distinct failure class that the other four cannot catch. The accompanying notebook implements the full pipeline, triggers four failure experiments by removing one wall at a time, and documents a Human Decision Node where AI-proposed configurations were rejected on architectural grounds.

**Master Argument:** Architecture is the leverage point, not the model.

**Core Claim:** A prompt that works in prototyping breaks silently in production because there is no versioning, no evaluation loop, no fallback when the LLM response degrades. The architecture causes the failure — not the model.

---

## Repository Structure

```
├── chapter-19-readme.txt                    # This file
├── Final Author's Note — Chapter 19.pdf     # 3-page Author's Note
├── Images/                                  # Figure Architect diagrams
├── production_prompt_pipeline.ipynb         # Jupyter notebook (runnable)
├── SubStack.md                              # Published Substack chapter
└── Youtube Link.docx                        # Video link
```

---

## Quick Start

```bash
# Clone the repo
git clone https://github.com/tanyabansal98/chapter19-production-prompt-pipeline.git
cd chapter19-production-prompt-pipeline

# Install dependencies
pip install pydantic numpy

# Open the notebook
jupyter notebook production_prompt_pipeline.ipynb
```

**No API keys required.** The notebook uses mock LLM responses with configurable drift modes (`normal`, `format_drift`, `semantic_drift`) so all five walls and all four failure experiments run out of the box.

---

## The Five Walls

| Wall | What It Catches | What Breaks Without It |
|------|----------------|----------------------|
| 1. Prompt Registry | Version mismatch | Cannot roll back a bad prompt; rollback means redeploying the entire app |
| 2. Schema Gate | Format drift | Invalid field values (e.g. `"URGENT"` instead of `"HIGH"`) flow into business logic |
| 3. Evaluation Loop | Semantic drift | Structurally valid but meaningfully wrong outputs pass all other checks silently |
| 4. Fallback Cascade | Total failure | A single API timeout becomes a user-facing 500 error |
| 5. Observability Layer | Invisible failure | Other walls still work but you cannot see what they are catching |

---

## Failure Experiments

The notebook includes four deliberate failure experiments. Each disables one wall and shows the result:

| Experiment | Wall Removed | What Happens |
|-----------|-------------|-------------|
| 1 | Schema Gate | Invalid enum values pass through to the application with no error |
| 2 | Evaluation Loop | Semantically degraded outputs pass schema validation — no alert, no exception |
| 3 | Fallback Cascade | Any single failure (timeout, rejection) crashes the pipeline |
| 4 | Observability | Walls still function but failures are invisible — no traces, no diagnostics |

**Experiment 2 is the critical one.** It demonstrates silent degradation: the output has all the right fields, correct types, and valid enum values, but the meanings are wrong. Without the evaluation loop, this goes undetected indefinitely.

---

## Human Decision Node

The AI scaffold proposed default configurations that were rejected on architectural grounds:

| AI Proposed | I Replaced With | Reasoning |
|------------|----------------|-----------|
| Uniform eval threshold (0.70) across all fields | Field-weighted scoring | `priority` controls ticket routing (high cost of error); `sentiment` is analytics-only (low cost). A uniform threshold treats them as equal when they are not. |
| Batch evaluation every 30 minutes | Inline evaluation on every response | A misrouted HIGH-priority ticket cannot wait 30 minutes for detection. |
| 2-hour cache TTL for fallback responses | 30-minute cache TTL | In customer support, a 2-hour-old cached response could reference a resolved issue. |

Each rejection traces to a specific failure the AI's default would have caused.

---

## Learning Outcomes

1. **Analyze** the architectural gap between a prototype prompt workflow and a production pipeline
2. **Design** a prompt management pipeline with version control, schema validation, and evaluation loops
3. **Evaluate** a deployed prompt system's health by detecting semantic drift and output degradation
4. **Implement** a fallback architecture that degrades gracefully under failure
5. **Critique** a prompt deployment by tracing the causal chain from a missing component to an observable failure

---

## Deliverables

| Deliverable | File |
|------------|------|
| Substack Chapter | `SubStack.md` |
| Jupyter Notebook | `production_prompt_pipeline.ipynb` |
| 10-Minute Video | `Youtube Link.docx` |
| Author's Note | `Final Author's Note — Chapter 19.pdf` |
| Figures | `Images/` |

---

## Tools Used

| Tool | Purpose |
|------|---------|
| Bookie the Bookmaker | Drafted chapter prose; ensured Tetrahedron (Structure–Logic–Implementation–Outcome) in every section |
| Eddy the Editor | Audited draft for Feynman Standard, architecture-without-mechanism gaps, and sycophantic AI usage |
| Eddy the Storyboarder | Generated scene-by-scene video storyboard with Explain → Show → Try structure |
| Figure Architect | Identified high-assertion zones and generated figure prompts for 6 publication-quality diagrams |

---

## Video

**Link:** https://youtu.be/I7S7XL_zLVc

Structure: Explain (2:30) → Show (5:30) → Try (2:00)

- **Explain:** States the architectural claim, draws the prototype vs production topology by hand on camera, names silent degradation as the failure mode
- **Show:** Walks through the notebook — prototype, five walls, Human Decision Node (3 rejections on camera), four failure experiments triggered live
- **Try:** Gives the viewer a specific exercise: disable the evaluation loop, enable semantic drift mode, observe silent degradation with no alerts

---

## Requirements

- Python 3.9+
- pydantic
- numpy

No external API keys or LLM access needed.

---

## License

Academic use — Northeastern University, INFO 7375, Spring 2026.
