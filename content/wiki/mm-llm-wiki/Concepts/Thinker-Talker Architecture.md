---
title: "Thinker-Talker Architecture"
type: concept
tags:
  - concept
  - speech
  - architecture
  - speech-generation
  - streaming
sources: 1
---

# Thinker-Talker Architecture

The speech-generation design behind [[Qwen3-Omni]] (introduced in Qwen2.5-Omni): split an
omni-modal model into a **Thinker** that generates **text** and a **Talker** that generates
**streaming speech tokens**, run as one end-to-end model sharing conversational history.

## The split

- **Thinker** — the reasoning/text LLM (an MoE Transformer in Qwen3-Omni, 30B-A3B). Consumes
  text + [[AuT (Audio Transformer)]] audio tokens + vision tokens; emits text.
- **Talker** — a smaller MoE Transformer (3B-A0.3B) that emits speech codec tokens, conditioned
  on the Thinker's high-dimensional **multimodal** features and the streamed text of the turn.

**Key decoupling (Qwen3-Omni):** the Talker conditions on the Thinker's *multimodal features
but **not** its text representations*. Rationale: discrete text tokens ≈ text embeddings
(information-equivalent), while multimodal conditioning is what's needed to keep prosody/timbre
coherent (e.g. speech translation). A side benefit: external modules (RAG, function-calling,
safety filters) can intervene on the Thinker's **text** before it is voiced, and Thinker vs
Talker can even take **separate system prompts** (response style vs audio style).

## Streaming speech stack

1. **Talker backbone** predicts one codec frame per step — codebook 0 via a linear head.
2. An **MTP (multi-token prediction)** module predicts the remaining **residual codebooks**
   (multi-codebook **RVQ** representation → richer voices/paralinguistics).
3. **Code2Wav**, a lightweight **causal ConvNet** vocoder, incrementally renders the waveform
   **frame-by-frame** at **12.5 Hz** — single-frame immediate synthesis, replacing the
   block-wise DiT vocoder of Qwen2.5-Omni for lower latency/FLOPs.

Result: end-to-end first-packet latency ~**234 ms** (audio).

## Contrast with the wiki's other speech-generation designs

- **vs [[Moshi]]** — Moshi keeps text and audio in **one** model via [[Inner Monologue]]
  (time-aligned text as a prefix to audio tokens) and [[Multi-Stream Audio Modeling]] (parallel
  user/model streams), decoding hierarchical codec tokens with an [[RQ-Transformer]]
  (Temporal + Depth Transformer). Thinker-Talker instead **separates** the text model from the
  speech model and uses MTP + a ConvNet vocoder. Both ultimately autoregress over **RVQ codec
  tokens at 12.5 Hz**.
- **vs [[Interaction Models]]** — those interleave [[Time-Aligned Micro-Turns]] in a single
  model rather than splitting Thinker/Talker.
- **vs [[Voxtral]]** — no speech generation at all ([[Speech Understanding]], audio→text).

## See also

- [[Qwen3-Omni]] · [[AuT (Audio Transformer)]] · [[Real-Time Interactive Speech Models]]
