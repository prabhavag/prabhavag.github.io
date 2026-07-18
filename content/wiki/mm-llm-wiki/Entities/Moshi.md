---
title: "Moshi"
type: entity
entity_type: model
developer: "[[Kyutai]]"
tags:
  - entity
  - model
  - speech
  - full-duplex
sources: 1
---

# Moshi

A **speech-text foundation model** and **[[Full-Duplex Spoken Dialogue|full-duplex]]** spoken
dialogue system by [[Kyutai]] (Sept 2024). The first real-time full-duplex spoken LLM —
theoretical latency **160 ms** (200 ms in practice), below the ~230 ms human conversational
average. Open-sourced (`github.com/kyutai-labs/moshi`).

Moshi casts dialogue as **speech-to-speech generation**, replacing the cascaded
ASR → text-LLM → TTS pipeline that causes multi-second latency, a text information bottleneck,
and rigid turn-taking.

## Architecture

![[moshi-fig1-overview.png]]
> *Figure 1 — Overview of Moshi: the Helium-backed [[RQ-Transformer]] generates time-aligned
> text + semantic + acoustic tokens for both the user and Moshi streams, with [[Mimi (neural audio codec)|Mimi]]
> tokenizing/detokenizing audio at 12.5 Hz.*

Three components, all designed for **streaming/causal** inference:

- **[[Helium]]** — 7B text-LLM backbone (reasoning + knowledge), trained from scratch on 2.1T
  tokens.
- **[[Mimi (neural audio codec)|Mimi]]** — neural audio codec turning 24 kHz audio into discrete
  tokens at **12.5 Hz / 1.1 kbps**, fusing semantic + acoustic information.
- **[[RQ-Transformer]]** — a large **Temporal Transformer** (across time, initialized from
  Helium) plus a small **Depth Transformer** (across the codebooks within one 80 ms step).

Two behaviours layered on top:

- **[[Multi-Stream Audio Modeling]]** — models **Moshi's** and the **user's** audio as two
  parallel token streams, removing explicit speaker turns (always listening + speaking).
- **[[Inner Monologue]]** — predicts time-aligned **text tokens as a prefix** to audio tokens;
  the biggest quality lever, and the knob that turns Moshi into a streaming ASR/TTS.

Joint sequence: **K = 2Q+1 = 17** sub-streams (1 text + 8 Moshi audio + 8 user audio), Q = 8.

## Training stages

1. **Helium pre-training** (text only, 2.1T tokens).
2. **Moshi pre-training** — Temporal Transformer warm-started from Helium; 50% of batches kept
   text-only to retain knowledge.
3. **Post-training** — simulated multi-stream from diarized audio.
4. **Fisher fine-tuning** — gains true full-duplex behaviour on real 2-speaker phone calls.
5. **Instruction fine-tuning** — on synthetic interaction scripts.

## Training data (how real-world data is used)

The recipe turns cheap, abundant real audio into the time-aligned, multi-stream supervision a
full-duplex model needs:

- **~7M hours of unlabeled real audio** (mostly English speech, 24 kHz mono) for pre-training.
  It has no transcripts, so **Whisper (large-v3) generates them** — ASR produces the aligned
  text stream ([[Inner Monologue]]). Single-stream at this stage.
- **Fisher — 2000 h of real two-channel phone calls** between paired strangers. Because each
  speaker is on a **separate channel**, it gives *ground-truth separated streams* to learn
  [[Multi-Stream Audio Modeling|listening + speaking at once]] (8 kHz → upsampled to 24 kHz).
- **PyAnnote diarization** over the 7M-hour set splits each recording into *main speaker* vs
  *residual* streams — synthesizing the two-stream structure from single-channel audio at scale
  (the post-training stage).
- **170 h of real multi-channel conversations** (natural + scripted, per-speaker channels) — used
  to train a realistic multi-stream TTS and fine-tune the backbone, not to train Moshi directly.
- **>20k h synthetic instruct speech** grounded in real **Wikipedia / StackExchange** text;
  Moshi's voice fixed to one actor's recordings, the "user" voice randomized for robustness.

**Transferable principle** (likely relevant to time-aligned models like
[[TML-Interaction-Small]], though TML doesn't disclose its data): scale from unlabeled real
audio + ASR-generated aligned text; learn overlap/interruption from real multi-channel
conversations; use diarization to bridge abundant single-channel audio to the multi-stream
format.

## Benchmark snapshot

| Task | Result | Note |
|---|---|---|
| Spoken QA (Web/Llama Questions, audio TriviaQA) | best speech-to-speech | Inner Monologue ~3× vs audio-only |
| Streaming TTS (LibriSpeech test-clean) | **4.7% WER** | beats Vall-E (5.9%); 2 s lookahead |
| Streaming ASR | **5.7% WER** | 80 ms alignment precision |
| MMLU | 49.7 | down from [[Helium]]'s 54.3 (cost of audio training) |

### Spoken question answering (Table 8, 0-shot accuracy %)

| Model | Web Q. | Llama Q. | Audio Trivia QA |
|---|---|---|---|
| _Audio only_ | | | |
| GSLM | 1.5 | 4.0 | – |
| AudioLM | 2.3 | 7.0 | – |
| TWIST (7B) | 1.1 | 0.5 | – |
| **Moshi** (w/o [[Inner Monologue]]) | **9.2** | **21.0** | 7.3 |
| _Text and audio_ | | | |
| SpeechGPT (7B) | 6.5 | 21.6 | 14.8 |
| Spectron (1B) | 6.1 | 22.9 | – |
| **Moshi** | **26.6** | **62.3** | **22.8** |
| Moshi (w/o text batches in pre-train) | 23.2 | 61.3 | 18.3 |
| _Text upper bound_ — [[Helium]] | 32.3 | 75.0 | 56.4 |

Inner Monologue ~triples accuracy over audio-only Moshi; the gap to text-only Helium (esp. on
Trivia QA) is the cost of oral-style fine-tuning.

## Relation to other work

- **Predecessor to [[Interaction Models]] / [[TML-Interaction-Small]]** (cited as full-duplex
  prior art). Contrast: Moshi is audio-only via a **codec**, single-model, fixed-rate token
  streams; TML is multimodal, **[[Encoder-Free Early Fusion|encoder-free]]**, with a
  **[[Interaction-Background Model Split|background model]]** and **[[Time-Aligned Micro-Turns]]**.
- **vs [[Voxtral]] (a [[Speech Understanding]] model).** Voxtral is audio→**text** only (no
  speech output, not real-time), so it can use a heavy batch **[[Whisper large-v3]] encoder +
  adapter** where Moshi commits to the streaming [[Mimi (neural audio codec)|Mimi]] codec.
  Convergences: both **ASR-pseudo-label** untranscribed audio, and both land on a **12.5 Hz**
  audio frame rate. Text↔audio coupling differs: Moshi's [[Inner Monologue]] (token-level text
  prefix) vs Voxtral's [[Audio-Text Pretraining Patterns]] (segment-level `<repeat>`/`<next>`).

## Safety

Voice consistency, toxicity/regurgitation analyses, and **watermarking** of generated audio.

Detailed in source: [[Moshi (Kyutai paper)]].
