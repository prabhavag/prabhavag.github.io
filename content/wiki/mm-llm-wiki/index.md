---
title: "Audio LLM Wiki"
type: index
updated: 2026-07-11
sources: 4
pages: 24
---

Catalog of every wiki page. Read this first when answering a query, then drill into the
relevant pages. Updated on every ingest.

## Sources
- [[Interaction Models (TML blog)]] — TML's research-preview announcement of native
  real-time interaction models (2026-05-10). Carries a `## Demo media` catalog (hero + 9
  capability videos + frame animation + benchmark clips).
  Original: [thinkingmachines.ai/blog/interaction-models](https://thinkingmachines.ai/blog/interaction-models/#capabilities).
- [[Moshi (Kyutai paper)]] — Kyutai's tech report (arXiv 2410.00037, Sept 2024) for the first
  real-time full-duplex speech-text dialogue model. Original: [arXiv 2410.00037](https://arxiv.org/pdf/2410.00037).
- [[Voxtral (Mistral paper)]] — Mistral's tech report (arXiv 2507.13264, July 2025) for Voxtral
  Mini & Small, open-weights audio-understanding (speech→text) models.
  Original: [arXiv 2507.13264](https://arxiv.org/pdf/2507.13264).
- [[Qwen3-Omni (technical report)]] — Qwen Team's tech report (arXiv 2509.17765, Sept 2025) for
  Qwen3-Omni, an omni-modal (text/image/audio/video) Thinker-Talker MoE with real-time speech.
  Original: [arXiv 2509.17765](https://arxiv.org/pdf/2509.17765).

## Entities
- [[TML-Interaction-Small]] — 276B MoE (12B active); first model strong on both intelligence
  and interactivity.
- [[FD-bench]] — interactivity benchmark suite (latency, quality, tool use).
- [[Moshi]] — Kyutai's real-time full-duplex speech-text dialogue model (160ms latency).
- [[Mimi (neural audio codec)]] — streaming neural audio codec (12.5Hz, 1.1kbps) powering Moshi.
- [[Helium]] — 7B text LLM backbone of Moshi, trained from scratch on 2.1T tokens.
- [[Voxtral]] — Mistral's open-weights audio-understanding models (Mini 4.7B / Small 24.3B); Whisper encoder → adapter → Mistral LLM, audio→text.
- [[Qwen3-Omni]] — Qwen Team's omni-modal model (30B-A3B MoE); text/image/audio/video in, text/speech out; ~234ms streaming, claims no modality degradation.
- [[AuT (Audio Transformer)]] — Qwen3-Omni's audio encoder; from-scratch on 20M hrs, 12.5Hz tokens, block-wise window attention for real-time prefill (replaces Whisper).

## Concepts
- [[Interaction Models]] — models that handle interaction natively instead of via a harness.
- [[Time-Aligned Micro-Turns]] — interleaved 200ms input/output stream chunks (core mechanism).
- [[Interaction-Background Model Split]] — real-time model + async background reasoning model.
- [[Encoder-Free Early Fusion]] — dMel / hMLP / flow-head multimodal I/O, co-trained from scratch.
- [[Full-Duplex Spoken Dialogue]] — listening and speaking at once; the problem space behind Moshi & Interaction Models.
- [[Multi-Stream Audio Modeling]] — parallel user + model token streams (Moshi); removes speaker turns.
- [[Inner Monologue]] — time-aligned text tokens as a prefix to audio tokens (Moshi); also yields streaming ASR/TTS.
- [[RQ-Transformer]] — Temporal + Depth Transformer for streaming hierarchical token modeling (Moshi).
- [[Speech Understanding]] — the audio-in/text-out paradigm (Voxtral); contrasted with full-duplex speech-to-speech.
- [[Audio-Text Pretraining Patterns]] — repetition (`<repeat>`) + cross-modal continuation (`<next>`); Voxtral's pretraining.
- [[Thinker-Talker Architecture]] — split of a text-generating Thinker + speech-token-generating Talker (Qwen3-Omni); MTP + Code2Wav streaming.

## Topics
- [[Real-Time Interactive Speech Models]] — synthesis comparing [[Moshi]],
  [[Interaction Models]], and [[Qwen3-Omni]] (the three real-time sources); also contrasts the
  understanding-only [[Voxtral]].
