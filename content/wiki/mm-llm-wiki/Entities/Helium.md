---
title: "Helium"
type: entity
entity_type: model
developer: "[[Kyutai]]"
tags:
  - entity
  - model
  - text-llm
sources: 1
---

# Helium

The **7B-parameter text language model** that serves as the reasoning/knowledge backbone of
[[Moshi]]. Trained from scratch by [[Kyutai]] on **2.1T tokens** of filtered English.

## Architecture & training

- Autoregressive Transformer: **RMSNorm**, **RoPE** positional embeddings, **4,096** context,
  FlashAttention, **Gated Linear Units** (SiLU gating).
- **32k** SentencePiece unigram tokenizer (digits split to single tokens, byte-backoff).
- AdamW, fixed LR then cosine decay.
- Data: 12.5% curated (Wikipedia, Wikibooks, StackExchange, peS2o scientific articles) + 87.5%
  CommonCrawl, heavily filtered (line-level dedup via bloom filter + fastText fuzzy dedup,
  fastText language ID, fastText 9-category quality classifier).

## Results

On-par with or better than similarly-compute-budgeted ~7B models (MPT, Falcon, Llama 2, OLMo)
across ARC, OBQA, HellaSwag, WinoGrande, PIQA, TriviaQA, NQ, MMLU; competitive even with
Mistral/Gemma (≈3× more training compute) on some. **MMLU 54.3** — drops to 49.7 once adapted
into [[Moshi]] (the cost of audio training).

**Table 2 — text LM evaluation** (ARCe/ARCc, OBQA, HellaSwag, WinoGrande, PIQA, SIQA,
TriviaQA Unfiltered/Wiki, NQ, MMLU; bold = best):

| Model | ARCe | ARCc | OBQA | HS | WG | PIQA | SIQA | TQA | NQ | MMLU |
|---|---|---|---|---|---|---|---|---|---|---|
| **Helium** | **79.6** | **55.9** | 53.6 | 76.3 | **70.0** | 79.4 | **51.0** | **59.9/72.6** | 23.3 | **54.3** |
| MPT | 70.5 | 46.5 | 51.4 | **77.6** | 69.9 | **80.6** | 48.5 | –/61.2 | 20.8 | 30.8 |
| Falcon | 73.7 | 47.5 | 53.0 | 76.3 | 68.9 | 80.3 | 47.2 | –/64.6 | 21.0 | 28.0 |
| Llama 2 | 75.2 | 45.9 | **58.6** | 77.2 | 69.2 | 78.8 | 48.3 | –/72.1 | **25.7** | 45.3 |
| OLMo | 67.2 | 42.5 | 50.0 | 75.5 | 69.8 | 77.5 | – | –/– | – | 52.0 |
| Mistral | 80.5 | 54.9 | 52.2 | 81.0 | 74.2 | 82.2 | 47.0 | 62.5/– | 23.2 | 62.5 |
| Gemma 1 | 81.5 | 53.2 | 52.8 | 81.2 | 72.3 | 81.2 | 51.8 | 63.4/– | 23.0 | 64.3 |

(Mistral & Gemma use ≈3× more training compute than Helium's 2.1T tokens.)

In [[Moshi]], Helium initializes the **Temporal Transformer** of the [[RQ-Transformer]]; 50%
text-only batches are retained during Moshi pre-training to preserve its knowledge.

Source: [[Moshi (Kyutai paper)]].
