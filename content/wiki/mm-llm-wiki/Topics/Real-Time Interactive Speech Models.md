---
title: "Real-Time Interactive Speech Models"
type: topic
tags:
  - topic
  - real-time
  - speech
  - full-duplex
sources: 3
---

# Real-Time Interactive Speech Models

Overview/synthesis of the wiki's recurring theme: models that **interact in real time** rather
than through a turn-based cascade. Three real-time sources so far ([[Moshi]],
[[Interaction Models]], [[Qwen3-Omni]]), all rejecting the ASR → text-LLM → TTS pipeline in
favour of making interactivity **native to the model** — plus [[Voxtral]] as an
understanding-only sibling.

## The shared thesis

- The cascade (VAD + ASR + LLM + TTS) is a **harness of less-intelligent components** — high
  latency, a text bottleneck that drops paralinguistic signal, and rigid speaker turns that
  can't represent overlap/interruption/backchannel.
- Fix: model the conversation as **continuous streams** so silence, overlap, and interruption
  stay in context → [[Full-Duplex Spoken Dialogue]].

## The approaches compared

| Axis           | [[Moshi]] ([[Kyutai]], 2024)                                                  | [[Interaction Models]] / [[TML-Interaction-Small]] (2026)        | [[Qwen3-Omni]] ([[Qwen Team]], 2025)                                                          |
| -------------- | ----------------------------------------------------------------------------- | ---------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Modalities     | audio ↔ audio (+ text scaffold)                                               | audio + video + text                                             | **omni**: text/image/audio/video in, text/speech out                                          |
| Duplex mode    | **full-duplex** — parallel per-speaker streams, models overlap/interruption   | **full-duplex** — 200 ms interleaved input/output ([[FD-bench]]) | **half-duplex** — native end-to-end streaming, but turn-based (no simultaneous listen+speak)¹ |
| Streaming unit | fixed-rate parallel token streams (12.5 Hz) — [[Multi-Stream Audio Modeling]] | 200 ms multimodal [[Time-Aligned Micro-Turns]]                   | per-frame codec at 12.5 Hz ([[Thinker-Talker Architecture]])                                  |
| Audio in       | codec ([[Mimi (neural audio codec)]])                                         | [[Encoder-Free Early Fusion]]                                    | **[[AuT (Audio Transformer)]]** encoder (20M-hr, from scratch)                                |
| Audio out      | discrete **codec** (Mimi, RVQ)                                                | encoder-free (dMel/flow head)                                    | multi-codebook **RVQ** + MTP + **Code2Wav** ConvNet                                           |
| Reasoning      | single model ([[Helium]] backbone)                                            | + async [[Interaction-Background Model Split]]                   | Thinker (MoE) + optional **Thinking** variant                                                 |
| Text coupling  | **[[Inner Monologue]]** (text prefix per step)                                | text co-trained in the fused model                               | Talker decoupled from Thinker **text** (uses multimodal features)                             |
| Latency        | 160 ms theoretical / 200 ms practical                                         | 0.40 s turn-taking ([[FD-bench]] V1)                             | 234 ms audio / 547 ms A-V first packet                                                        |
| Openness       | open-source weights                                                           | research preview                                                 | open-weights (30B-A3B MoE)                                                                    |

¹ [[Qwen3-Omni]] is the cluster's **non-full-duplex** member: it rejects the ASR→LLM→TTS
**cascade** (one native, end-to-end model) and streams responses at low latency, but the
[[Qwen3-Omni (technical report)|report]] never claims full-duplex — no interruption/overlap/
turn-taking modeling, and its chunked *perceive → Thinker → Talker* pipeline is turn-based. So
it kills the cascade for **latency**, not for **turn-taking**. ("Half-duplex" is inferred from
the architecture and the absence of any full-duplex claim, not an authors' label.)

## Convergences & divergences

- **Converge on:** no speaker turns; streaming-first design; keeping silence/overlap in
  context; treating the "harness" as the thing to eliminate. Notably, both [[Moshi]] and
  [[Qwen3-Omni]] land on **RVQ codec speech at 12.5 Hz** (~80 ms/token) — a recurring
  real-time design point.
- **Diverge on:** codec vs encoder-free I/O; single fused model ([[Inner Monologue]]) vs a
  **text/speech split** ([[Thinker-Talker Architecture]]) vs interaction+background split;
  audio-only vs full multimodal; fixed-rate tokens vs micro-turns; adapting Whisper vs a
  from-scratch **[[AuT (Audio Transformer)]]** encoder.

## A sibling paradigm: speech *understanding*

[[Voxtral]] (Mistral, 2025) is **not** a real-time/full-duplex model and sits outside the table
above — it's the contrasting category. It takes **audio → text** (transcribe, translate, QA,
summarize); it does **not** generate speech or converse in real time. It still rejects the classic
cascade, but for a different end: where Moshi / Interaction Models kill the cascade for
**latency & turn-taking**, Voxtral folds the ASR step into the LLM for **deeper comprehension**
of audio content. It also takes a third stance on the audio path — a **[[Whisper large-v3]]
encoder + adapter** (vs Moshi's codec, vs TML's [[Encoder-Free Early Fusion]]). See
[[Speech Understanding]] for the full paradigm contrast.

## Open threads

- Run [[Moshi]] and [[Qwen3-Omni]] on [[FD-bench]] / proactivity benchmarks for an
  apples-to-apples comparison.
- [[Inner Monologue]] (text prefix, one model) vs [[Thinker-Talker Architecture]] (text/speech
  split) vs TML text co-training — which scales better, and what does decoupling text cost?
- Does the codec route ([[Mimi (neural audio codec)]], multi-codebook RVQ) cap
  quality/extensibility vs encoder-free?
- Could a [[Speech Understanding]] encoder+adapter stack ([[Voxtral]]) be made streaming/full-duplex,
  or is the batch encoder a hard ceiling? (Qwen3-Omni's [[AuT (Audio Transformer)]] answers part
  of this with block-wise window attention for real-time prefill.)
- Qwen3-Omni claims **no modality degradation** from joint training — does that hold up against
  dedicated speech-only systems on turn-taking/full-duplex, not just benchmark accuracy?

## Sources

- [[Interaction Models (TML blog)]] · [[Moshi (Kyutai paper)]] · [[Qwen3-Omni (technical report)]] · [[Voxtral (Mistral paper)]] (understanding-only sibling)
