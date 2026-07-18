---
title: "Encoder-Free Early Fusion"
type: concept
tags:
  - concept
  - architecture
  - multimodal
sources: 1
---

# Encoder-Free Early Fusion

The multimodal input/output design of [[TML-Interaction-Small]]. Rather than routing audio
and video through large standalone encoders/decoders (e.g. a Whisper-like encoder or a
separate TTS model), it uses **minimal preprocessing** with all components **co-trained from
scratch** alongside the transformer:

| Modality | Handling |
|---|---|
| Audio in | **dMel** representation → lightweight embedding layer |
| Image/video in | split into **40×40 patches**, encoded by an **hMLP** |
| Audio out | **flow head** (flow-matching decoder) |
| Text | standard embedding / unembedding |

Each 200ms micro-turn the model takes any subset of {text, audio, frame} and predicts text
and mel. Co-training avoids the quality/latency overhead and modularity seams of bolt-on
encoders, fitting the "interactivity is part of the model" thesis of [[Interaction Models]].

### Figure — single-micro-turn architecture (source Fig. 4)

![[fig4-architecture.png]]

> *An illustration of the interaction model architecture for a single 200ms micro-turn. The
> model takes in any subset of text, audio, or video and predicts text and audio.*

## Contrast: codec-based tokenization (Moshi)

The opposite design bet is [[Moshi]]'s [[Mimi (neural audio codec)|Mimi]] codec: a learned
discrete tokenizer/detokenizer (RVQ) that audio is routed through. TML deliberately **avoids**
such standalone encoders/decoders, feeding minimally-preprocessed signals co-trained from
scratch. Both target real-time multimodal I/O; they disagree on whether to commit to a codec.
See [[Moshi (Kyutai paper)]].

## Contrast: encoder + adapter (Voxtral)

The **third** design point is [[Voxtral]] (Mistral), which does exactly what TML avoids: a large
standalone **[[Whisper large-v3]] encoder** (640M, pretrained ASR) feeding a downsampling
**adapter** into a frozen-ish Mistral LLM. This is the *encoder-based* end of the spectrum —
mature, off-the-shelf audio features at the cost of the modularity seams and the encoder's fixed
30 s receptive field. Three bets on getting audio into a transformer: **encoder+adapter**
(Voxtral) · **codec** (Moshi/Mimi) · **encoder-free co-training** (TML). Note Voxtral is
*understanding-only* (audio→text), so it can afford a heavy batch encoder where the real-time
models cannot. See [[Speech Understanding]].

Source: [[Interaction Models (TML blog)]].
