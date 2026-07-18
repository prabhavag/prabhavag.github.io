---
title: "Interaction-Background Model Split"
type: concept
tags:
  - concept
  - architecture
  - agentic
sources: 1
---

# Interaction-Background Model Split

### Figure — system overview (source Fig. 2)

![[fig2-system-overview.png]]

> *The user continuously interacts with the interaction model, while the background model
> performs asynchronous tasks. Both systems share their context.*

The two-model system architecture used by [[TML-Interaction-Small]]:

- **Interaction model** — maintains real-time presence via [[Time-Aligned Micro-Turns]];
  perceives and responds continuously.
- **Background model** — runs **asynchronously** for sustained reasoning, tool use, and
  longer-horizon agentic work.

When a task needs deeper reasoning than can be produced instantly, the interaction model
**delegates** by sending a rich context package (the full conversation, not a standalone
query). Results stream back and the interaction model **interleaves** them into the
conversation at a moment appropriate to what the user is currently doing — avoiding an abrupt
context switch. The interaction model stays present throughout (answering follow-ups, taking
new input, holding the thread).

## Payoff

Responsiveness of a non-thinking model **and** the planning/tool-use of a reasoning model.
Both halves are independently intelligent — the interaction model alone is competitive on
intelligence benchmarks.

Demos (from the source) — full catalog on [[Interaction Models (TML blog)]]:

**Simultaneous tool calls and search**
<iframe width="100%" height="315" src="https://www.youtube.com/embed/ly3GtaiRFyo" title="Simultaneous tool calls and search" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

**Longer real session**
<iframe width="100%" height="315" src="https://www.youtube.com/embed/qXdYDUqxSxA" title="Longer real session" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

## Open questions

- How does this delegation compare to other agentic/orchestration architectures?
- The post calls the background agents "early-stage" — room for the two to cooperate more
  deeply.

## Contrast: single-model full-duplex (Moshi)

[[Moshi]] is a counterpoint — a **single** streaming model with no background/reasoning split.
It gets responsiveness from architecture ([[Multi-Stream Audio Modeling]], a fast codec) rather
than by offloading heavy reasoning. The split is TML's bet for "thinking-model intelligence at
non-thinking latency"; Moshi shows full-duplex is achievable without it (at lower reasoning
depth). See [[Moshi (Kyutai paper)]].

Source: [[Interaction Models (TML blog)]].
