---
title: "TML-Interaction-Small"
type: entity
entity_type: model
developer: "[[Thinking Machines Lab]]"
tags:
  - entity
  - model
  - multimodal
sources: 1
---

The interaction model released as a research preview by [[Thinking Machines Lab]]. Presented
as the first model with **both** strong intelligence/instruction-following **and**
interactivity.

Overview video — Introducing interaction models:

<iframe width="100%" height="315" src="https://www.youtube.com/embed/A12AVongNN4" title="Introducing interaction models" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

Capability demos and benchmark clips catalogued on [[Interaction Models (TML blog)]].

## Architecture

- **276B-parameter Mixture-of-Experts, 12B active.**
- Natively multimodal (audio, video, text) via [[Time-Aligned Micro-Turns]] — 200ms
  interleaved input/output chunks.
- [[Encoder-Free Early Fusion]]: dMel audio input, hMLP over 40×40 image patches, flow-head
  audio decoder; all components co-trained from scratch.
- Pairs with an asynchronous background model — see [[Interaction-Background Model Split]].

## Benchmark snapshot

| Benchmark | Result | Note |
|---|---|---|
| [[FD-bench]] V1 turn-taking latency | **0.40s** | best of all models compared |
| FD-bench V1.5 average | **77.8** | next best ~54 (Gemini min) |
| FD-bench V3 response quality / pass@1 | 82.8 / 68.0 | with background agent enabled |
| Audio MultiChallenge APR | 43.4% | best *instant* model; < GPT-2.0 xhigh (48.5%) |
| IFEval (text) | 89.7% | — |
| Harmbench refusal rate | 99.0% | — |

## Limitations

Larger pretrained models from the same family are currently too slow to serve in this
real-time regime; long A/V sessions strain context management; needs reliable connectivity.

## Training data

TML **does not disclose** the training corpus. The blog says only that the model is
**trained from scratch** on continuous audio + video + text, and that capabilities "improve in
quality as we scale up model size and **training data**" (the bitter-lesson bet). The only
data specifics are for **safety**, and they're synthetic: a TTS model generates colloquial
refusal / over-refusal examples, and an automated red-teaming harness generates multi-turn
refusal data to keep speech refusals in parity with the text model.

So *how real-world data trains the core interaction behaviour is unknown* for TML-small. The
closest **documented** analogue is [[Moshi]] (see its "Training data" section): unlabeled real
audio + ASR-generated aligned text, real two-channel conversations (Fisher) for full-duplex,
and diarization to synthesize multi-stream structure — principles that plausibly transfer,
with TML additionally needing time-aligned **video** that Moshi's audio-only recipe doesn't cover.

## Compared to Moshi

[[Moshi]] ([[Kyutai]], 2024) is the closest predecessor — also real-time and
[[Full-Duplex Spoken Dialogue|full-duplex]], but **audio-only**, a **single** model (no
[[Interaction-Background Model Split|background split]]), and built on a discrete
[[Mimi (neural audio codec)|codec]] rather than [[Encoder-Free Early Fusion]]. TML-Interaction-Small
extends the regime to **audio + video + text** with stronger intelligence. See
[[Moshi (Kyutai paper)]].

Detailed in source: [[Interaction Models (TML blog)]].
