---
title: "RQ-Transformer"
type: concept
tags:
  - concept
  - architecture
  - speech
sources: 1
---

The hierarchical, streaming architecture [[Moshi]] uses to model many interleaved token
sub-streams per timestep. It factorizes generation into two Transformers:

![[moshi-fig3-rq-transformer.png]]
> *Figure 3 — The RQ-Transformer breaks a flattened length-K·S sequence into S timesteps for a
> large Temporal Transformer (producing a context embedding `z_s`) that conditions a smaller
> Depth Transformer over the K sub-streams. (K = 4 shown for illustration.)*


- **Temporal Transformer** — the large model, runs **across time steps** (one step = 80 ms /
  one [[Mimi (neural audio codec)|Mimi]] frame). Initialized from [[Helium]].
- **Depth Transformer** — a small model (dim 1024, 6 layers) that runs **across the sub-streams
  within a single step** (the K codebooks/text), predicting them bottom-to-top.

This split lets one big model capture long-range temporal structure while a cheap inner model
resolves the per-step codebook hierarchy — keeping the whole thing **streaming-capable**.

## Notable design choices

- **Per-codebook parameters** in the Depth Transformer.
- **Acoustic delay** — offsetting acoustic tokens by 1–2 steps behind semantic tokens reduces
  inter-codebook dependence at a step and markedly improves generation stability/quality.
- Generates **semantic and acoustic tokens jointly** in a streaming fashion (vs prior work that
  produced all semantic tokens first), and extends naturally to the
  **[[Multi-Stream Audio Modeling|multi-stream]]** + **[[Inner Monologue]]** joint sequence
  (K = 17 sub-streams).

Source: [[Moshi (Kyutai paper)]].
