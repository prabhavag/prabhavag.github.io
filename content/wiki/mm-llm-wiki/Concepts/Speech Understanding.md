---
title: "Speech Understanding"
type: concept
tags:
  - concept
  - speech
  - multimodal
  - architecture
sources: 1
---

# Speech Understanding

The paradigm of **audio-in → text-out** multimodal LLMs: a model that *comprehends* spoken
audio (transcribe, translate, answer questions about it, summarize, call functions) and
responds in **text**. Distinct from the wiki's **full-duplex speech-to-speech** cluster
([[Moshi]], [[Interaction Models]]), which *generate speech* and converse in real time.
[[Voxtral]] is the wiki's canonical speech-understanding model.

## The architecture pattern

Speech-understanding models typically bolt an **audio encoder + adapter** onto a pretrained
text LLM:

```
audio → [audio encoder] → [adapter / projector] → audio embeddings ┐
                                                                    ├→ [text LLM decoder] → text
text prompt ─────────────────────────→ text embeddings ────────────┘
```

- The **encoder** (e.g. [[Whisper large-v3]] in Voxtral) turns waveform into embeddings.
- An **adapter** downsamples them (Voxtral: 50 Hz → 12.5 Hz, 4×) so they fit the decoder's
  context and roughly match a text token's information density.
- The **LLM decoder** (Voxtral: [[Ministral 3B]] / [[Mistral Small 3.1]]) treats audio
  embeddings as just more input tokens and generates text.

This is **early fusion via an encoder** — contrast [[Encoder-Free Early Fusion]] (TML, no
standalone encoder) and **codec tokenization** ([[Moshi]]'s [[Mimi (neural audio codec)]],
discrete audio tokens). Three different bets on how audio enters a transformer.

## Understanding ≠ transcription

A central finding (from [[Voxtral]]'s ablations and its motivation): **transcription and
understanding are separable**, and the field historically over-indexed on transcription.

- Transcription/translation are well-served by ASR models (Whisper, etc.); the harder,
  under-evaluated tasks are **QA, reasoning, and summarization over audio**, especially
  long-context (up to ~40-min files in Voxtral's 32K window).
- In [[Voxtral]] the two come from **different training signals** — see
  [[Audio-Text Pretraining Patterns]] (repetition → transcription; cross-modal continuation →
  understanding).

## Evaluation gap

Voxtral argues the speech-eval ecosystem lacked breadth/standardization (mostly WER/BLEU). It
contributes **speech-synthesized GSM8K / TriviaQA / MMLU** and an internal **SU benchmark**
(LLM-as-judge over audio QA) to push evaluation toward reasoning. (Compare the full-duplex
cluster's interactivity-first benchmark, [[FD-bench]] — different axis entirely: latency &
turn-taking, not comprehension.)

## Where it sits relative to full-duplex dialogue

| | **Speech Understanding** ([[Voxtral]]) | **Full-Duplex Dialogue** ([[Moshi]], [[Interaction Models]]) |
|---|---|---|
| Output modality | text only | speech (+ text scaffold) |
| Real-time / streaming | batch, up to 40-min files | yes, sub-second latency |
| Turn structure | request/response (text query over audio) | continuous, overlap-aware ([[Full-Duplex Spoken Dialogue]]) |
| Audio path | encoder + adapter into a frozen-ish LLM | codec / encoder-free, co-trained |
| Goal | *comprehend* audio | *converse* in audio |

Both reject the classic cascade, but for different ends: full-duplex kills the cascade for
**latency/turn-taking**; speech-understanding folds the ASR step into the LLM for **deeper
comprehension** of the audio's content.

The audio-**in** path of an omni-modal model like [[Qwen3-Omni]] is essentially a
speech-understanding stack too — its [[AuT (Audio Transformer)]] encoder feeds audio tokens to
the Thinker (audio→text), the same shape as Voxtral's Whisper+adapter, before the Talker adds
speech *generation* on top.

Source: [[Voxtral (Mistral paper)]].
