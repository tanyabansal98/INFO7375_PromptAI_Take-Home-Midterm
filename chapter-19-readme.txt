# Chapter 19: Building Production Prompt Systems — From Prototype to Pipeline

A publication-ready chapter for *Design of Agentic Systems with Case Studies: Prompt Engineering with LLMs*, demonstrating that a prompt working in a prototype will fail silently in production without versioning, evaluation, fallback, and observability. The chapter introduces a "Five Walls" architecture — Prompt Registry, Schema Gate, Evaluation Loop, Fallback Cascade, and Observability Layer — each blocking a distinct failure class that the other four cannot catch. The accompanying notebook implements the full pipeline, triggers four failure experiments by removing one wall at a time, and documents a Human Decision Node where AI-proposed configurations were rejected on architectural grounds.

## Repository Structure

```
├── README.md
├── chapter-19-prose.md              # Full Substack chapter (revised draft)
├── notebook/
│   ├── production_prompt_pipeline.py # Jupyter notebook (percent format)
│   └── requirements.txt
├── figures/
│   ├── fig1-prototype-vs-production.md
│   ├── fig2-pipeline-architecture.md
│   ├── fig3-failure-cascade-chains.md
│   ├── fig4-operational-loop.md
│   ├── fig5-structural-vs-semantic.md
│   └── fig6-fallback-tiers.md
├── authors-note.pdf
└── video-link.md
```

## Quick Start

```bash
git clone https://github.com/tanyabansal98/Chapter19-Production-Prompt-Pipeline.git
cd Chapter19-Production-Prompt-Pipeline/notebook

pip install -r requirements.txt

# Option A: Run in JupyterLab / VS Code (recognizes percent-format cells)
# Open production_prompt_pipeline.py and run cells interactively

# Option B: Convert to .ipynb first
pip install jupytext
jupytext --to notebook production_prompt_pipeline.py
jupyter notebook production_prompt_pipeline.ipynb
```

No API keys required. The notebook uses mock LLM responses with configurable drift modes (`normal`, `format_drift`, `semantic_drift`) so all five walls and all four failure experiments run out of the box.

## Key Architectural Claim

**Architecture is the leverage point, not the model.** The same GPT-4 model produces reliable or unreliable results depending entirely on whether the pipeline includes evaluation gates, schema validation, and fallback logic. The model does not change across the failure experiments — the architecture does.

## Failure Mode

A prompt that works in prototyping breaks silently in production because the architecture has no mechanism to detect, contain, or recover from the model's inevitable variability. Specifically: when the evaluation gate is removed, semantically degraded outputs (structurally valid but wrong in meaning) pass through every remaining check — schema validation, fallback logic, observability — with no alert, no error, and no exception. Detection time without the evaluation loop: weeks, if ever.

## Human Decision Node

The AI scaffold proposed a uniform evaluation threshold of 0.70 across all output fields with batch-mode scoring and a 2-hour cache TTL. Three rejections were documented:

1. **Uniform threshold → field-weighted scoring.** The `priority` field controls ticket routing (high cost of error); `sentiment` is analytics-only (low cost of error). A single threshold treats them as equal when they are not.
2. **Batch evaluation → inline evaluation.** A misrouted HIGH-priority ticket discovered 30 minutes later in a batch review is operationally unacceptable for this use case.
3. **2-hour cache TTL → 30-minute TTL.** Support tickets are time-sensitive; a stale cached summary could reference a since-resolved issue.

The full decision log with architectural reasoning is in `notebook/production_prompt_pipeline.py`, Part 3.

## Video

📹 **[Video Link]** *(replace with YouTube/Vimeo URL before submission)*

10-minute walkthrough following the Explain → Show → Try structure:
- **Explain (2:30):** Architectural claim, prototype vs. production topology, failure mode
- **Show (5:30):** Notebook demo, pipeline walkthrough, Human Decision Node on camera, failure experiment triggered live
- **Try (2:00):** Exercise for the viewer, open question on scale-dependent failure modes

## Course Context

**Course:** INFO 7375 — Prompt Engineering for Generative AI
**Institution:** Northeastern University, Khoury College of Computer Sciences
**Term:** Spring 2026
**Assignment:** Take-Home Midterm — The Agentic Author's Mandate
**Chapter Type:** Type E — Operational / Production Practice

## Author

**Tanya Bansal**
MS in Information Systems, Northeastern University
[GitHub](https://github.com/tanyabansal98) · [LinkedIn](https://linkedin.com/in/tanyabansal98)
