---
title: "Interaction Models: A Scalable Approach to Human-AI Collaboration"
type: source
source: "https://thinkingmachines.ai/blog/interaction-models/"
author:
  - "[[Thinking Machines Lab]]"
published: 2026-05-10
created: 2026-06-02
tags:
  - source
  - interaction-models
  - multimodal
  - real-time
---

# Interaction Models: A Scalable Approach to Human-AI Collaboration

**Source clip:** [[Interaction Models A Scalable Approach to Human-AI Collaboration]] · [[Thinking Machines Lab]] · published 2026-05-10

Research-preview announcement of **[[Interaction Models]]** — models that handle interaction
*natively* rather than through external scaffolding. The thesis: interactivity should scale
*with* intelligence, so it must be part of the model itself rather than bolted on with a
harness ("the bitter lesson" applied to interaction).

## Key takeaways

- **The collaboration bottleneck.** Turn-based interfaces experience reality in a single
  thread: the model can't perceive the user mid-turn, and its perception freezes while it
  generates. This narrow channel pushes humans out of the loop. Interaction models aim to
  make AI "interactive in real time across any modality."
- **[[TML-Interaction-Small]]** is the released model — a 276B-parameter MoE (12B active),
  claimed as the first model strong on *both* intelligence and interactivity.
- **[[Time-Aligned Micro-Turns]]** are the core mechanism: continuous audio/video/text
  streams split into interleaved 200ms input/output chunks, so silence, overlap, and
  interruption stay in context. No artificial turn boundaries; no VAD harness.
- **[[Interaction-Background Model Split]]:** a real-time interaction model stays present
  while delegating heavy reasoning/tool-use to an asynchronous background model, weaving
  results back in when contextually appropriate. Buys "thinking-model intelligence at
  non-thinking latency."
- **[[Encoder-Free Early Fusion]]:** minimal preprocessing — dMel for audio, hMLP over 40×40
  image patches, a flow head for audio decode — all co-trained from scratch.
- **Inference**: 200ms chunks need frequent small prefills; they built *streaming sessions*
  (chunks appended to a persistent GPU sequence), upstreamed to SGLang. Also gather+gemv MoE
  kernels and batch-invariant kernels for bitwise trainer-sampler alignment.

## Capabilities unlocked

Seamless dialog management (no separate component), verbal/visual interjections, simultaneous
speech (e.g. live translation), time-awareness, and concurrent tool calls / search /
generative UI while listening and speaking.

## Results

- Dominates interaction quality on **[[FD-bench]]** v1.5 (77.8 avg vs ~46–54 for baselines)
  and best responsiveness on FD-bench V1 (0.40s turn-taking latency).
- Competitive intelligence: Audio MultiChallenge APR 43.4% (beats other *instant* models;
  below thinking GPT-realtime-2.0 xhigh at 48.5%). IFEval text 89.7%.
- New internal benchmarks for proactivity where no existing model performs: **TimeSpeak**,
  **CueSpeak** (time/simultaneous speech) and adapted **RepCount-A**, **ProactiveVideoQA**,
  **Charades** (visual proactivity).

## Limitations / future work

Long sessions accumulate context fast (active work); low-latency A/V streaming needs reliable
connectivity; larger pretrained models are currently too slow to serve in this regime
(larger releases planned later in the year); background agents seen as early-stage.

## Open threads to pursue

- How does the background-agent handoff compare to other agentic-delegation architectures?
- Reproducibility of the FD-bench / Audio MultiChallenge numbers (some baselines self-reported
  via Scale AI / Artificial Analysis).
- Relationship of this real-time regime to [[Interaction Models]] as a broader research area.

## Demo media

The blog page is heavily illustrated; the web clipper dropped nearly all of it, so the media
is restored here as embedded players (YouTube `<iframe>`s — this vault's image-localizer
mangles `![](url)` embeds, so HTML iframes are used). Also restored into the raw clip.

**Hero — Introducing interaction models**
<iframe width="100%" height="315" src="https://www.youtube.com/embed/A12AVongNN4" title="Introducing interaction models" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

**Capability demos** (under `#capabilities`):

**Seamless dialog management**
<iframe width="100%" height="315" src="https://www.youtube.com/embed/Ys6i_MGnjUA" title="Seamless dialog management" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

**Verbal interjection**
<iframe width="100%" height="315" src="https://www.youtube.com/embed/_SsogUUZP2o" title="Verbal interjection" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

**Visual interjection**
<iframe width="100%" height="315" src="https://www.youtube.com/embed/n2GXGjy41HQ" title="Visual interjection" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

**Simultaneous speech**
<iframe width="100%" height="315" src="https://www.youtube.com/embed/2ky5MXBvZP8" title="Simultaneous speech" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

**Time awareness**
<iframe width="100%" height="315" src="https://www.youtube.com/embed/sqJNfze19jA" title="Time awareness" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

**Simultaneous tool calls and search**
<iframe width="100%" height="315" src="https://www.youtube.com/embed/ly3GtaiRFyo" title="Simultaneous tool calls and search" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

**Generative UI**
<iframe width="100%" height="315" src="https://www.youtube.com/embed/GL1waQJsV9c" title="Generative UI" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

**Longer real session**
<iframe width="100%" height="315" src="https://www.youtube.com/embed/qXdYDUqxSxA" title="Longer real session" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

**Continuous collaboration**
<iframe width="100%" height="315" src="https://www.youtube.com/embed/iVDJ8O89ERg" title="Continuous collaboration" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

**"Our approach" figures** — four diagrams (CSS/SVG/animated on the page, no static image
files), captured as **screenshots** (`raw/assets/fig{1..4}-*.png`) and embedded on the
relevant concept pages:
1. *Turn-based vs. time-aligned timeline* → `fig1-timeline.png`, on [[Time-Aligned Micro-Turns]].
2. *System overview* (user ↔ interaction model ↔ background model) → `fig2-system-overview.png`,
   on [[Interaction-Background Model Split]].
3. *Human perception vs. single interleaved token sequence* → `fig3-token-timeline.png`,
   on [[Time-Aligned Micro-Turns]].
4. *Single 200ms micro-turn architecture* (dMel / hMLP / flow head) → `fig4-architecture.png`,
   on [[Encoder-Free Early Fusion]]; the source's inline SVG is also preserved in the raw clip.

**Benchmark example clips** (New dimensions of interactivity) — 5 audio-comparison examples,
each comparing input vs our model vs baselines:
`https://thinkingmachines.ai/audio/interaction-models/example-{1..5}/{input,our_model,gpt_realtime_2,gpt_realtime_2_thinking_xhigh,gemini_think_high,gemini_think_minimal}.wav`.
Only example-5 adds a video:

<video width="100%" height="315" controls preload="metadata" src="https://thinkingmachines.ai/audio/interaction-models/example-5/video.mp4"></video>
