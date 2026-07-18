---
title: "Mimi (neural audio codec)"
type: entity
entity_type: model
developer: "[[Kyutai]]"
tags:
  - entity
  - model
  - audio-codec
  - speech
sources: 1
---

# Mimi (neural audio codec)

The **streaming neural audio codec** that tokenizes audio for [[Moshi]]. Mimi is the piece
that makes real-time speech-to-speech possible: it turns 24 kHz waveforms into discrete tokens
a language model can predict, and back, **causally** (low-latency, streaming).

![[moshi-fig2-mimi.png]]
> *Figure 2 — Mimi's SeaNet encoder/decoder with a Transformer bottleneck and **split RVQ**: a
> semantic VQ (distilled from a frozen WavLM via cosine similarity) in parallel with a 7-level
> acoustic RVQ, trained with adversarial losses.*

## Key specs

- **24 kHz** input → latent at **12.5 frames/sec**, dim 512 (SeaNet conv autoencoder, all
  **causal** convolutions; 80 ms frame size and stride).
- **Residual Vector Quantization (RVQ):** Q = 8 quantizers, codebook size 2048 → **1.1 kbps**.
- **Transformer bottleneck** (8 layers, causal) before and after quantization for quality.
- **Adversarial-only training** (feature + discriminator loss, no reconstruction loss) — a
  counter-intuitive but large subjective quality win.
- Quantizer dropout for bitrate scalability; quantization applied only 50% of the time during
  training (improves quality, more so at low bitrate).

## Split RVQ — fusing semantic + acoustic tokens

Audio LMs usually need **semantic** tokens (linguistic, from a self-supervised model) *and*
**acoustic** tokens (high-fidelity reconstruction). Computing both separately is non-causal
and expensive. Mimi instead **distills** non-causal **WavLM** embeddings into its tokens via a
**split RVQ**: a single **semantic VQ** in parallel with a **7-level acoustic RVQ**, summed.
This avoids forcing acoustic detail into the residual of the semantic quantizer, giving a
better semantic/acoustic trade-off while staying **streaming-compatible**.

## Why it matters

Mimi is the [[Moshi]] design choice that most directly **contrasts** with TML's
[[Encoder-Free Early Fusion]]: Moshi commits to a learned discrete **codec** (tokenizer +
detokenizer), whereas [[TML-Interaction-Small]] avoids standalone codecs/encoders and feeds
minimally-preprocessed signals (dMel/hMLP) co-trained from scratch. Same problem (real-time
multimodal I/O), opposite bet on tokenization.

Source: [[Moshi (Kyutai paper)]].
