# Stakeholder-Facing RAG Deployment & Business Telemetry Platform

> **Role**: Lead AI Platform & Analytics Engineer · Entrance-Exam EdTech Enterprise

## The Problem

Building a performant RAG pipeline is a technical achievement. Getting institutional buy-in to deploy it in production is a different problem entirely. Executives, operations leads, and academic administrators do not evaluate systems on BLEU scores or cosine similarity distributions — they evaluate on **unit economics, operational SLAs, and risk**.

Without a compelling, real-time stakeholder demonstration that translates AI quality metrics into business outcomes — tutor deflection rates, cost savings, latency budgets, escalation compliance — even a technically excellent pipeline stalls in a proof-of-concept limbo indefinitely.

## The Solution

This project packages a benchmarked reference-free RAG quality evaluation pipeline into an **interactive, stakeholder-facing dual-tab demonstration application** — purpose-built to close the gap between AI engineering and executive decision-making.

### Application Architecture

**Tab 1 — Live Student Doubt Resolution Simulator**

Demonstrates the pipeline in action across three live scenarios:
- **Syllabus calculation queries** — structured, high-confidence resolution with source citations
- **Conceptual understanding queries** — nuanced, context-grounded explanation generation
- **Out-of-scope edge cases** — demonstrates controlled, transparent hallucination interception and safe escalation

Each resolution surfaces real-time confidence gauges and retrieved source citations, making the system's decision-making transparent and auditable.

**Tab 2 — Executive Business & Operations Telemetry**

Presents live unit economics and operational KPIs:
- Real-time tutor deflection rate
- Projected monthly cost savings vs. human tutoring benchmarks
- Latency profiling across the full inference pipeline
- Financial telemetry aggregation for CFO-level reporting

### Technical Components

- **Modular Inference Engine** — Dense curriculum vector retrieval integrated with real-time proposition entailment scoring
- **RAG Quality Index (RQI) Decision Gate** — Automated pass/escalate routing with configurable threshold
- **In-Flight Latency Profiling** — End-to-end latency measured per query component (retrieval, generation, scoring)
- **Financial Telemetry** — Per-query and aggregate cost modelling against human tutoring unit economics

### Tech Stack

- **Embeddings**: SentenceTransformers · ChromaDB
- **Inference**: Python modular pipeline with real-time scoring
- **Dashboard**: Streamlit (dual-tab stakeholder interface)
- **Telemetry**: In-process latency profiling · financial aggregation
- **Notebook**: JupyterLab (fully executed, with live telemetry integration)

## Measurable Business Impact

| Metric | Result |
|---|---|
| Tutor deflection rate | **44.5%** of queries resolved autonomously without human tutor involvement |
| Projected monthly cost savings | **$20,760** vs. equivalent human tutoring coverage |
| Latency | Real-time — sub-second resolution for auto-approved queries |
| Out-of-scope hallucination rate | **0%** — all fringe queries safely escalated, none reaching students |
| Stakeholder interface | Dual-tab: live resolution demo + executive financial telemetry |
| Deployment readiness | Full SLA compliance with educational data security standards |
| Strategic outcome | Transforms the RAG pipeline from an engineering prototype into a board-ready business case |

The **44.5% tutor deflection rate translating to $20,760 in monthly savings** is the number that moves a pilot into a production contract. This application exists to make that number visible, credible, and interactive — so stakeholders can stress-test it, adjust assumptions, and arrive at the deployment decision with confidence.

## Repository Contents

| File | Description |
|---|---|
| `starter.ipynb` | Starter notebook with scaffolding |
| `final output 1.ipynb` | Fully executed inference notebook (retrieval + RQI gate + latency + telemetry) |
| `final output 2.csv` | Query-level telemetry log (resolution outcomes, latency, cost per query) |
| `final output 3.py` | Stakeholder-facing dual-tab Streamlit application |
| `artifact 1.py` | Modular inference engine |
| `artifact 2.csv` | Financial telemetry aggregation export |
| `artifact 3–6.png` | Executive dashboard visualisations |
| `input 1.csv` | Curriculum knowledge base |
| `input 2.csv` | Incoming student doubt evaluation set |
| `start_dashboard.py` | Streamlit launcher |
| `master_recording_guide.md` | Full deployment walkthrough |
