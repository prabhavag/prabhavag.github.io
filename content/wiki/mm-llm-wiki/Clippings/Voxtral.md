---
title: "Voxtral"
source: "https://arxiv.org/html/2507.13264v1"
author:
published:
created: 2026-07-11
description:
tags:
  - "clippings"
---
###### Abstract

We present Voxtral Mini and Voxtral Small, two multimodal audio chat models. Voxtral is trained to comprehend both spoken audio and text documents, achieving state-of-the-art performance across a diverse range of audio benchmarks, while preserving strong text capabilities. Voxtral Small outperforms a number of closed-source models, while being small enough to run locally. A 32K context window enables the model to handle audio files up to 40 minutes in duration and long multi-turn conversations. We also contribute three benchmarks for evaluating speech understanding models on knowledge and trivia. Both Voxtral models are released under Apache 2.0 license.

![[Uncaptioned image]](https://arxiv.org/html/2507.13264v1/images/header.jpeg)

<table><tbody><tr><td>Webpage:</td><td><a href="https://mistral.ai/news/voxtral/">https://mistral.ai/news/voxtral/</a></td></tr><tr><td rowspan="2">Model weights:</td><td><a href="https://huggingface.co/mistralai/Voxtral-Mini-3B-2507">https://huggingface.co/mistralai/Voxtral-Mini-3B-2507</a></td></tr><tr><td><a href="https://huggingface.co/mistralai/Voxtral-Small-24B-2507">https://huggingface.co/mistralai/Voxtral-Small-24B-2507</a></td></tr><tr><td>Evals:</td><td><a href="https://huggingface.co/collections/mistralai/speech-evals-6875e9b26c78be4a081050f4">https://huggingface.co/collections/mistralai/speech-evals-6875e9b26c78be4a081050f4</a></td></tr></tbody></table>

## 1 Introduction

This paper describes Voxtral Mini and Voxtral Small, a pair of multimodal language models trained to understand both speech and text, released with open-weights under an Apache 2.0 license. Voxtral is pretrained on a large-scale corpus of audio and text documents, and subsequently instruction tuned on real and synthetic data. It is capable of responding directly to audio (or text) and answering questions about audio files. With a 32K token context window, Voxtral is capable of processing audio files up to 40 minutes long.

Compared with similarly sized models in the same evaluation setting, we find that Voxtral delivers strong audio reasoning capabilities without sacrificing text-only performance. Its performance is state-of-the-art for speech transcription and translation, outperforming other open-weights and closed models. In speech question-answering (QA) and summarization, it performs comparably with closed models of a similar price class, such as GPT-4o mini [^14] and Gemini 2.5 Flash [^5].

During evaluation of Voxtral and other models, we found that the existing ecosystem of speech evaluations lacked breadth and standardization; the majority of previous work focused on evaluation of transcription and translation quality, and less on other understanding tasks. In Section 3.4, we present evaluations that measure a wider range of speech comprehension and reasoning tasks.

Our primary contributions are:

- Two open-weights audio models with state-of-the-art transcription and multilingual speech understanding for audio durations up to their 32K context window
- Native function calling support with audio
- Evaluation benchmarks that measure speech understanding and reasoning

The report is structured as follows: First, we outline our modeling choices. Next, we describe methods for pretraining, post-training, and response quality enhancement. Finally, we present benchmark results along with architectural and data ablations.

## 2 Modeling

Voxtral is based on the Transformer architecture [^26], consisting of three components: an audio encoder to process speech inputs, an adapter layer to downsample audio embeddings, and a language decoder to reason and generate text outputs. The overall architecture is depicted in Figure 1.

![Refer to caption](https://arxiv.org/html/2507.13264v1/images/voxtral-arch.002.jpeg)

Figure 1: Voxtral Architecture. The audio encoder processes the speech input, attending to 30-second chunks of audio independently. The audio embeddings are concatenated at the output, and downsampled by a factor of 4x in the audio-language adapter. The multimodal LLM decoder auto-regressively predicts text tokens, conditional on the audio and text inputs.

### 2.1 Audio Encoder

The audio encoder is based on Whisper large-v3 [^24]. A raw audio waveform is first mapped to a log-Mel spectrogram [^7] with 128 Mel-bins and 160 hop-length. Within the Whisper encoder, the spectrogram passes through a convolutional stem that downsamples its temporal resolution by a factor of two, after which it is fed into a stack of bidirectional self-attention layers. The resulting audio embeddings have a frame rate of 50 Hz.

Whisper has a fixed receptive field of 30 seconds. To accommodate audio sequences exceeding this duration, we compute the log-Mel spectrogram for the entire audio, but restrict the encoder to independently process each 30 second chunk. The absolute positional encodings are reset for each chunk, and chunks from the same audio are partitioned into a batch axis. Within the encoder’s attention layers, this approach is functionally equivalent to chunk-wise attention [^30], which mitigates the computational overhead for longer audio inputs and enhances length generalization. The embeddings computed from each chunk are concatenated at the output stage, forming a unified representation of the complete audio sequence.

Due to its fixed receptive field, Whisper also pads short audios to 30 seconds. In Section 5.1, we investigate removing this padding requirement to allow continuous audio lengths. However, empirical results demonstrated a decline in performance, even when tuning the encoder to adapt. Consequently, we maintain the practice of padding all audio inputs to the next multiple of 30 seconds.

### 2.2 Adapter Layer

The high frame-rate of the audio encoder would result in long sequence-lengths through the language decoder. For example, a 30 minute audio at 50Hz has a sequence length of 90k tokens, leading to high memory and slow inference. To circumvent this, we append an additional MLP layer at the audio encoder outputs that is responsible for downsampling the audio embeddings. In Section 5.2, we show a downsampling factor of 4x yields the best trade-off between sequence-length and performance. This results in an effective frame-rate of 12.5Hz, enabling Voxtral to gracefully handle audios up to 40 minutes with a context-length of 32k tokens.

### 2.3 Language Decoder

We release two variants of Voxtral: Mini and Small. Voxtral Mini is built on top of Ministral 3B [^17], an edge-focused model that delivers competitive performance with a small memory footprint. Voxtral Small leverages the Mistral Small 3.1 24B backbone [^18], giving strong performance across a range of knowledge and reasoning tasks. Table 1 decomposes the number of parameters in each checkpoint based on the sub-components.

Table 1: Parameter Counts. Number of parameters for Voxtral Mini and Small.

|  | Audio Encoder | Audio Adapter | Text Embeddings | Language Decoder | Total |
| --- | --- | --- | --- | --- | --- |
| Mini | 640M | 25M | 400M | 3.6B | 4.7B |
| Small | 640M | 52M | 670M | 22.9B | 24.3B |

## 3 Methodology

We train the model in three phases: pretraining, supervised finetuning, and preference alignment. Each phase is described separately below. Finally, we describe our evaluation protocol for speech understanding tasks.

### 3.1 Pretraining

The pretraining stage of Voxtral is designed to introduce speech to the language decoder, complementary to the existing modality of text. Given an audio dataset with text transcriptions, we first chunk the audio into short segments together with their corresponding transcription, forming parallel audio-text pairs: $(A_{1},T_{1}),(A_{2},T_{2}),\dots,(A_{N},T_{N})$. The segmentation boundaries are defined by upstream voice activity detection and diarization models. If transcripts are unavailable, we pseudo-label the audio with an ASR model.

Similar to prior works [^20] [^29], we define two patterns that combine audio and text into training samples for the model: audio-to-text repetition and cross-modal continuation.

The audio-to-text repetition pattern is defined as an audio segment $A_{n}$ followed by the corresponding transcription $T_{n}$. A training sample consists of a single audio-text pair $(A_{n},T_{n})$. This formulation mimics speech recognition and is used to explicitly teach the model speech-to-text alignment.

On the other hand, the cross-modal continuation pattern is designed to implicitly align the speech and text modalities through modality-invariant context modeling. Specifically, for each audio segment $A_{n}$, the corresponding text is the proceeding text segment in the sequence $T_{n+1}$. In addition, a training sample is composed by interleaving audio and text for multiple consecutive segments: $(A_{1},T_{2},A_{3},T_{4},\dots,A_{N-1},T_{N})$. This structure resembles tasks like QA or conversation, where the model must maintain discourse continuity across modalities.

Since we use two different data patterns, the proceeding text segment for a given audio segment is ambiguous; both repeat and continuation are valid. To eliminate ambiguity, we introduce two special tokens to specify the expected output: \<repeat> for repetition and \<next> for continuation. These tokens are used for pattern indication during training and as part of the prompt during inference to control model behavior.

The two pretraining patterns are shown with their special tokens in Figure 2. Note that we treat each audio-transcription pair as a standalone sequence wrapped with \<bos>/\<eos> without previous context.

![Refer to caption](https://arxiv.org/html/2507.13264v1/images/PT.png)

Figure 2: Pretraining patterns. A single audio-text example ( A, T ) (A,T) is first segmented into a set of audio-text pairs { n } = 1 N \\left\\{(A\_{n},T\_{n})\\right\\}\_{n=1}^{N}, based on the timestamps and transcriptions returned by segmentation stage. For the audio-to-text repetition pattern, a given audio A\_{n} is repeated in the text space T\_{n}. For the cross-modal continuation pattern, each audio is followed by its subsequent text + T\_{n+1}. The task is signaled to the model by the and special tokens respectively.

During pretraining, we balance the two patterns evenly. We demonstrate in Section 5.3 that this balanced approach is essential; the audio-to-text repetition pattern drives transcription performance, while the cross-modal continuation pattern prepares the model for speech understanding tasks that require deeper reasoning and context integration, such as audio-based QA or dialogue. To preserve text capabilities, we also include text pretraining data in the data mixture.

For the first pass over the data mixture, we freeze the audio encoder and language decoder, training only the adapter. We found this warm-up stage beneficial for speech understanding evaluations, whereas speech recognition results are similar with and without warm-up. We also perform one pretraining run on the Mini scale with just the audio-to-text repetition pattern. We call this model Voxtral Mini Transcribe, and compare it to other ASR-only models in Section 4.1.

### 3.2 Supervised Finetuning

In post-training, our primary objective is to preserve or slightly enhance the transcription capabilities established during pretraining, while simultaneously extending the model’s proficiency over a range of speech understanding tasks. We also develop robust instruction-following behavior, irrespective of whether user inputs are in audio or text form.

Our speech understanding data falls into two categories: tasks where audio is provided as context and the assistant responds to text queries, and tasks where the assistant responds directly to audio inputs. Both categories rely significantly on synthetic data.

#### Audio Context, Text Query

To create synthetic data for tasks involving audio context paired with text queries, we utilize long-form audio data (segments up to approximately 40 minutes) with corresponding transcripts and language identification metadata. Transcripts are paired with tailored prompts and fed into an LLM (Mistral Large), which then generates question-answer pairs related to the audio content. The prompts explicitly instruct the LLM to frame both questions and answers as though they arise from auditory comprehension rather than text analysis, thereby encouraging natural responses from the downstream audio model. To achieve data diversity and richness, we vary question types, including straightforward factual inquiries, "needle-in-haystack" retrieval tasks, and reasoning-intensive problems. Moreover, to minimize repetitive question styles, the LLM generates multiple candidate question-answer pairs per audio segment, from which we sample a single pair for inclusion in the post-training dataset. While we typically ensure that the question-answer pairs match the language of the original audio and transcript, we occasionally instruct Mistral Large to produce pairs in different languages to enable QA for audios in languages the user does not speak.

Additionally, we allocate another portion of the long-form audio data for synthetic summarization and translation data. For translation tasks, we leverage language identification metadata to select a target language different from the original audio language. To mitigate overfitting to a narrow range of user message patterns, we sampled from a large, manually curated set of plausible user requests.

#### Audio-Only Input

For scenarios in which the user provides only audio input, we adapt existing text supervised finetuning data, including function calling datasets, by converting text user messages into synthetic audio using a text-to-speech (TTS) model. However, reliance solely on TTS-generated audio leads to poor generalization to genuine human speech, particularly accented voices, manifesting most commonly in erroneous transcription of conversational prompts rather than appropriate continuation. To address this limitation, we extract questions from long-form ASR data that can be adequately answered through general world knowledge, thus requiring no additional audio context. We then isolate audio excerpts containing these questions and generate corresponding text answers using Mistral Large. This process yields datasets consisting of genuine human speech questions paired with text answers.

Speech recognition is a distinctive use case characterized by an unambiguous task, rendering the text prompt redundant. To address this, we introduce a dedicated "transcribe mode," signaled via a new special token. This mode explicitly instructs the model to perform transcription tasks, thereby eliminating the need for a text prompt.

### 3.3 Preference Alignment

Direct Preference Optimization (DPO) [^25] offers a lightweight alternative to full RLHF by learning directly from pairwise preferences. We also adopt its online variant [^12], where for each example, we sample two candidate responses from the current policy with temperature $T{=}0.5$. To rank responses, we take the entire conversation, replace the audio with its transcription, and leverage a text-based reward model. Although the reward model only has access to the audio transcription - rather than the raw audio itself - it is able to capture semantics, style, and factual coherence from this information, attributes that transfer to the generated text response. Our Online DPO implementation utilizes the sampling and reward infrastructure that powered the Magistral [^16] series.

We apply DPO and Online DPO to both Voxtral Mini and Small, for which we present results in Section 5.4. While both DPO and Online DPO helped improve the response quality, the online variant was more effective.

### 3.4 Evaluation

In addition to standard benchmarks for speech transcription, translation, and understanding - detailed in Sections A.1 and A.2 - we create our own test sets. These sets build upon existing research and evaluate model attributes that are typically underrepresented, particularly long-context QA.

#### Speech-Synthesized Benchmarks

To evaluate spoken-language understanding, prior works take existing text benchmarks and synthesize the text prompt into speech [^19] [^3]. We extend these test suites by creating speech-synthesized versions of three established text benchmarks: GSM8K [^4], TriviaQA [^15], and MMLU [^13].

The first step in creating these benchmarks involves filtering to only include prompts that are viable as speech-synthesized inputs, similar to [^10]. For every example, we classify it into one of three categories with Mistral Large:

- Verbalizable: plain wording or simple numerals. No re-write necessary.
- Verbalizable with Rewrite: math, code, or symbols, that can be deterministically rewritten into speech‑friendly text. For example, digits are converted to spelled-out form, acronyms expanded, markdown removed. The specific prompt used to achieve this is outlined in Appendix A.3.
- Non‑Verbalizable: text that cannot be naturally spoken, such as tables, figures or lengthy math and code, is discarded.

Once the valid set of examples is established, we synthesize each one individually using a TTS engine. To ensure diversity in speakers, we randomly sample speaker embeddings from a diverse set, trimmed to six-second clips and filtered to only include single-speaker utterances. For each prompt, we sample a speaker embedding from this pool and generate the corresponding audio input using the TTS engine. Since the model output is in the text-space, scoring the generations requires no additional modifications.

We are releasing the synthesized evaluations under a permissive license and encourage their adoption as standard benchmarks for speech understanding.

#### Speech Understanding (SU) Benchmark

We develop an internal benchmark that measures the ability of models to answer questions about audios in a helpful manner. The audio files range up to 19 minutes in duration, assessing understanding on moderately long audio contexts. We use an LLM as a judge, which has access to a transcription of the audio, the question, a reference answer, and the proposed answer. The LLM judge returns two complementary metrics:

1. llm\_judge\_score: a *binary helpfulness indicator*. The score is 1 if the answer is deemed correct and helpful to the user’s question, 0 otherwise.
2. grade\_llm\_judge\_score: a *0–5 quality grade*. A score of 0 means the answer is completely wrong, unhelpful, and poorly written; 5 denotes that it is factually correct, well-reasoned, and clearly presented. Intermediate values reflect partial correctness, clarity, and overall usefulness, as instructed in the grading prompt.

During evaluation, we independently judge each answer multiple times to capture sampling variability. The judge prompts are provided in A.4.

## 4 Results

We evaluate Voxtral on a range of speech recognition, translation, speech understanding, speech function calling and text benchmarks. We compare the model to GPT-4o mini Audio (/Transcribe) and Gemini 2.5 Flash, as well as Scribe and Whisper large-v3 on speech recognition tasks.

### 4.1 Speech Recognition

Figure 3 plots the macro‑averaged word error rates (WER) on four benchmarks: English Short‑Form, English Long‑Form, Mozilla Common Voice 15.1 (MCV) [^1] and FLEURS [^6]. We compute the macro‑average across tasks for English Short and Long-Form, and languages for MCV and FLEURS.

Voxtral Small achieves state-of-the-art transcription results on English Short-Form and MCV, beating all open and closed-source models. Voxtral Mini Transcribe performs competitively with much larger closed-source models, surpassing GPT-4o mini Transcribe and Gemini 2.5 Flash across all tasks. A full breakdown of English and multilingual word error rates are provided in A.1.

![Refer to caption](https://arxiv.org/html/2507.13264v1/results/multi_avg.png)

Figure 3: Speech Recognition Benchmarks. Macro-average WER results across tasks. Voxtral Small outperforms all open and closed-source models on English Short-Form and MCV. Voxtral Mini Transcribe beats GPT-4o mini Transcribe and Gemini 2.5 Flash in every task.

### 4.2 Speech Translation

We evaluate Voxtral on the FLEURS Speech Translation benchmark. We show BLEU scores for a subset of source/target pairs in Figure 4. Voxtral Small achieves state-of-the-art translation scores in every source/target combination.

![Refer to caption](https://arxiv.org/html/2507.13264v1/results/fleurs_translate.png)

Figure 4: FLEURS Translation. BLEU scores for source/target language pairs on the FLEURS Translation benchmark. Voxtral Small achieves state-of-the-art for every combination of languages.

### 4.3 Speech Understanding

We evaluate Voxtral on a range of public Speech QA benchmarks, such as Llama QA [^19] and Openbook QA [^3], as well as the speech-synthesized subsets of standard Text Understanding benchmarks. We also evaluate on our in-house speech understanding (SU) benchmark, consisting of in-the-wild audio examples with challenging QA-style prompts. Figure 5 highlights that Voxtral Small performs competitively with closed-source models, beating GPT-4o mini Audio on three of the seven tasks.

![Refer to caption](https://arxiv.org/html/2507.13264v1/results/audio_understanding.png)

Figure 5: Speech Understanding Benchmarks. We report the accuracy across three speech understanding benchmarks and three synthesized speech subsets of text benchmarks. Voxtral Small is competitive with closed-source models, surpassing GPT-4o mini Audio on three of the seven benchmarks.

### 4.4 Text Benchmarks

Figure 6 compares the performance of Voxtral Mini and Small to the text-only Mistral Small 3.1 model. Voxtral Small maintains performance across text-benchmarks, making it a suitable drop-in replacement for both text and audio tasks.

![Refer to caption](https://arxiv.org/html/2507.13264v1/results/text_with_voxtral_mini.png)

Figure 6: Text-Only Benchmarks. We report the accuracy across five standard text understanding benchmarks. Voxtral Small performs comparably to Mistral Small 3.1, highlighting its strong text capabilities.

## 5 Analysis

In this Section, we share results and analyses for two architectural ablations, the pretrain pattern format, and improvements from Online DPO.

### 5.1 To Pad or Not To Pad

Whisper pads short audios to 30-seconds. We investigate whether this padding constraint is necessary during pre-training, under the setting that the encoder weights are trained in order to adapt to the new configuration.

Figure 8 plots a subset of ASR and speech understanding results for models trained with and without padding. Disabling padding incurs almost no penalty on FLEURS English, however there is a 0.5% WER degradation on French. The 3-Shot Accuracy on Llama QA is comparable over the course of training for the two runs. To achieve the best possible speech recognition scores without compromise to speech understanding, we opt to maintain padding in the audio encoder.

![Refer to caption](https://arxiv.org/html/2507.13264v1/x1.png)

Figure 7: Effect of Padding. Word error rate results on FLEURS English (left) and FLEURS French (middle), alongside 3-shot Accuracy on Llama QA (right) for models trained with and without 30-second padding.

### 5.2 Adapter Downsampling

The baseline audio encoder operates at a frame-rate of 50 Hz. To reduce decoder computation and memory, we insert an MLP adapter layer that downsamples the audio embeddings along the temporal axis. We experiment with target frame‐rates of 50, 25, 12.5 and 6.25 Hz, corresponding to downsampling factors of 1x, 2x, 4x and 8x.

Figure 8 plots the WER on FLEURS English and French, as well as 3-Shot Accuracy on Llama QA. For 25 and 12.5 Hz, there is little degradation on ASR benchmarks. However, for 6.25 Hz, there is a penalty of over 1% on FLUERS French. On Llama QA, 12.5 Hz surpasses the 50 Hz baseline, achieving a score 1.5% higher. We hypothesize that at 12.5 Hz, each audio-embedding encodes a similar amount of information as a text-embedding in the language decoder backbone, leading to superior understanding performance. Based on the trade-off between sequence-length, ASR and speech-understanding performance, we select 12.5 Hz as the optimal frame-rate for Voxtral.

![Refer to caption](https://arxiv.org/html/2507.13264v1/x2.png)

Figure 8: Effect of Downsampling. Word error rate results on FLEURS English (left) and FLEURS French (middle), alongside 3-shot Accuracy on Llama QA (right) for various frame-rates, achieved by increasing the downsampling factor by powers of 2.

### 5.3 Pre-Training Patterns

Recall that we leverage two data patterns during pretraining: audio-to-text repetition and cross-modal continuation. Figure 9 demonstrates how changing the ratio of these two patterns affects ASR and speech understanding. To better understand the underlying capabilities of the cross-modal continuation pattern for ASR, we evaluate it on the 3-Shot version of the FLEURS ASR task, which is more aligned with the multi-turn pattern presented during training.

Including just the audio-to-text repetition pattern results in strong ASR performance, at the expense of nearly zero-performance on Llama QA. Conversely, training on just the cross-modal continuation pattern yields strong Llama QA performance, but a WER of nearly 60% on ASR. Balancing the two tasks with equal ratios achieves ASR and Llama QA performance comparable to the runs with a single pattern. Thus, we sample each pattern with equal probability during pretraining.

![Refer to caption](https://arxiv.org/html/2507.13264v1/x3.png)

Figure 9: Pattern Proportions. Word error rate results on FLEURS English (left) and FLEURS French (middle), alongside 3-shot Accuracy on Llama QA (right) for varying proportions of pretrain patterns.

### 5.4 DPO and Online DPO

Table 2 shows the LLM Judge and Grade scores on the SU Benchmark for the Voxtral SFT, DPO and Online DPO checkpoints. Each answer is independently judged ten times and we report the mean ± standard deviation.

For both Mini and Small, DPO and Online DPO improve response quality metrics relative to the SFT baselines. Qualitative inspection—including informal “vibe checks”—shows that the Voxtral Mini Online DPO variant delivers crisper grounding, fewer hallucinations, and generally more helpful responses, so we are releasing it as the public Voxtral Mini checkpoint.

For Voxtral Small, we saw substantial gains in response quality score as measured by the Speech Understanding Benchmark, but they are accompanied by a slight regression on the English short-form benchmarks. Hence, the default checkpoint remains the SFT model. We aim to release an Online DPO Voxtral Small model which does not regress on those ASR metrics in the near future.

Table 2: Response Improvements with Online DPO. Response quality on the internal SU benchmark (mean ± SD over ten trials), as well as the macro-average WER on the English short-form test sets. The differences in scores for other tasks were not significant. Hence, we omit them from this table. Note that GPT-4o mini Audio does not support transcription.

| Model | % LLM Judge $\uparrow$ | Grade $\uparrow$ | En Short WER $\downarrow$ |
| --- | --- | --- | --- |
| Voxtral Mini SFT | 83.47 ± 2.17 | 3.92 ± 0.04 | 6.77 |
| Voxtral Mini Offline DPO | 84.91 ± 3.21 | 3.92 ± 0.08 | 6.78 |
| Voxtral Mini Online DPO | 85.59 ± 3.77 | 4.08 ± 0.07 | 6.79 |
| Voxtral Small SFT | 86.61 ± 0.96 | 4.16 ± 0.03 | 6.31 |
| Voxtral Small Offline DPO | 87.29 ± 1.65 | 4.19 ± 0.04 | 6.32 |
| Voxtral Small Online DPO | 88.31 ± 2.03 | 4.38 ± 0.06 | 6.50 |
| GPT-4o mini Audio | 80.00 ± 2.97 | 3.97 ± 0.05 | \- |
| Gemini 2.5 Flash | 88.64 ± 2.28 | 4.54 ± 0.07 | 8.04 |

## 6 Conclusion

This paper presented Voxtral Mini and Voxtral Small, a pair of open-weights audio chat models. It demonstrated their capabilities in understanding spoken audio and text, both on existing and new benchmarks. Their strengths across a wide array of speech tasks, strong instruction following, and multilingual prowess make them highly versatile for complex multimodal tasks. Both models are released under the Apache 2.0 license.

### Core contributors

Alexander H. Liu, Andy Ehrenberg, Andy Lo, Clément Denoix, Corentin Barreau, Guillaume Lample, Jean-Malo Delignon, Khyathi Raghavi Chandu, Patrick von Platen, Pavankumar Reddy Muddireddy, Sanchit Gandhi, Soham Ghosh, Srijan Mishra, Thomas Foubert

### Contributors

Abhinav Rastogi, Adam Yang, Albert Q. Jiang, Alexandre Sablayrolles, Amélie Héliou, Amélie Martin, Anmol Agarwal, Antoine Roux, Arthur Darcet, Arthur Mensch, Baptiste Bout, Baptiste Rozière, Baudouin De Monicault, Chris Bamford, Christian Wallenwein, Christophe Renaudin, Clémence Lanfranchi, Darius Dabert, Devendra Singh Chaplot, Devon Mizelle, Diego de las Casas, Elliot Chane-Sane, Emilien Fugier, Emma Bou Hanna, Gabrielle Berrada, Gauthier Delerce, Gauthier Guinet, Georgii Novikov, Guillaume Martin, Himanshu Jaju, Jan Ludziejewski, Jason Rute, Jean-Hadrien Chabran, Jessica Chudnovsky, Joachim Studnia, Joep Barmentlo, Jonas Amar, Josselin Somerville Roberts, Julien Denize, Karan Saxena, Karmesh Yadav, Kartik Khandelwal, Kush Jain, Lélio Renard Lavaud, Léonard Blier, Lingxiao Zhao, Louis Martin, Lucile Saulnier, Luyu Gao, Marie Pellat, Mathilde Guillaumin, Mathis Felardos, Matthieu Dinot, Maxime Darrin, Maximilian Augustin, Mickaël Seznec, Neha Gupta, Nikhil Raghuraman, Olivier Duchenne, Patricia Wang, Patryk Saffer, Paul Jacob, Paul Wambergue, Paula Kurylowicz, Philomène Chagniot, Pierre Stock, Pravesh Agrawal, Rémi Delacourt, Romain Sauvestre, Roman Soletskyi, Sagar Vaze, Sandeep Subramanian, Saurabh Garg, Shashwat Dalal, Siddharth Gandhi, Sumukh Aithal, Szymon Antoniak, Teven Le Scao, Thibault Schueller, Thibaut Lavril, Thomas Robert, Thomas Wang, Timothée Lacroix, Tom Bewley, Valeriia Nemychnikova, Victor Paltz, Virgile Richard, Wen-Ding Li, William Marshall, Xuanyu Zhang, Yihan Wan, Yunhao Tang

## References

## Appendix A Appendix

### A.1 Speech Recognition - Full Results

Table 3 shows a task-breakdown of short-form English speech recognition results for LibriSpeech Test Clean [^22], LibriSpeech Test Other, GigaSpeech [^2], VoxPopuli [^28], SwitchBoard [^11], CHiME-4 [^27] and SPGISpeech [^21]. For English long-form, we take the one-hour long earnings calls from Earnings-21 [^8] and Earnings-22 [^9], and segment them into shorter, 10 minute variants. This is required to ensure that the full audio fits in a transcription request payload to closed-source providers.

Table 3: English speech recognition results by task. We report short-form scores for LibriSpeech Test Clean (LS-C), LibriSpeech Test Other (LS-O), GigaSpeech (GS), VoxPopuli (VP), SwitchBoard (SB), CHiME-4 (C-4) and SPGISPeech (SPGI). We report long-form scores for Earnings-21 10m (E21 10m) and Earnings-22 10m (E22 10m).

<table><tbody><tr><td></td><td colspan="7">Short-Form</td><td colspan="2">Long-Form</td></tr><tr><td>Model</td><td>LS-C</td><td>LS-O</td><td>GS</td><td>VP</td><td>SB</td><td>C-4</td><td>SPGI</td><td>E21 10m</td><td>E22 10m</td></tr><tr><td>Whisper large-v3</td><td>1.84</td><td>3.66</td><td>11.60</td><td>9.58</td><td>13.14</td><td>10.88</td><td>3.15</td><td>9.88</td><td>13.07</td></tr><tr><td>GPT4o mini Transcribe</td><td>1.92</td><td>4.70</td><td>14.80</td><td>7.34</td><td>17.31</td><td>11.35</td><td>4.51</td><td>10.09</td><td>12.27</td></tr><tr><td>Gemini 2.5 Flash</td><td>2.97</td><td>6.15</td><td>10.99</td><td>7.84</td><td>9.57</td><td>14.79</td><td>4.00</td><td>8.09</td><td>10.80</td></tr><tr><td>ElevenLabs Scribe</td><td>1.80</td><td>3.44</td><td>10.52</td><td>6.95</td><td>10.62</td><td>8.35</td><td>3.16</td><td>7.39</td><td>9.16</td></tr><tr><td>Voxtral Mini</td><td>1.86</td><td>4.04</td><td>10.68</td><td>6.85</td><td>11.32</td><td>10.59</td><td>2.19</td><td>9.62</td><td>11.84</td></tr><tr><td>Voxtral Mini Transcribe</td><td>1.57</td><td>3.21</td><td>10.04</td><td>6.78</td><td>11.35</td><td>10.03</td><td>2.04</td><td>9.52</td><td>12.18</td></tr><tr><td>Voxtral Small</td><td>1.53</td><td>3.14</td><td>10.27</td><td>6.62</td><td>11.09</td><td>9.64</td><td>1.89</td><td>9.55</td><td>12.48</td></tr></tbody></table>

Tables 4, 5 and 6 show the per-language breakdown of WER scores for the FLEURS, Mozilla Common Voice and Multilingual LibriSpeech [^23] benchmarks, respectively.

Table 4: Per-language WER scores for FLEURS Arabic, Dutch, English, French, German, Hindi, Italian, Portuguese and Spanish.

| Model | ar | nl | en | fr | de | hi | it | pt | es |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Whisper large-v3 | 15.44 | 5.87 | 4.00 | 5.55 | 5.46 | 28.87 | 2.71 | 3.90 | 2.81 |
| GPT-4o mini Transcribe | 14.02 | 5.54 | 3.19 | 4.51 | 3.76 | 12.36 | 2.02 | 3.54 | 2.58 |
| Gemini 2.5 Flash | 25.25 | 6.20 | 4.64 | 6.17 | 4.74 | 6.76 | 2.21 | 4.23 | 3.17 |
| Scribe | 11.58 | 4.63 | 3.29 | 5.07 | 4.78 | 5.67 | 1.48 | 4.50 | 3.13 |
| Voxtral Mini | 25.40 | 6.27 | 3.77 | 4.87 | 4.40 | 9.26 | 2.51 | 3.76 | 3.52 |
| Voxtral Mini Transcribe | 14.64 | 4.89 | 3.61 | 4.22 | 3.54 | 10.32 | 2.31 | 3.57 | 2.75 |
| Voxtral Small | 13.44 | 4.94 | 3.35 | 4.03 | 3.38 | 7.69 | 2.62 | 3.79 | 2.72 |

Table 5: Per-language WER scores for MCV Arabic, Dutch, English, French, German, Hindi, Italian, Portuguese and Spanish. For fairness, we omit Arabic from the macro-average in Figure 3, since all models score in excess of 45%.

| Model | ar | nl | en | fr | de | hi | it | pt | es |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Whisper large-v3 | 50.58 | 5.83 | 22.91 | 11.33 | 6.25 | 46.75 | 6.81 | 7.17 | 5.66 |
| GPT-4o mini Transcribe | 51.68 | 6.47 | 13.15 | 10.75 | 6.72 | 39.06 | 6.16 | 9.95 | 6.10 |
| Gemini 2.5 Flash | 53.62 | 5.73 | 12.31 | 11.86 | 7.25 | 11.09 | 6.64 | 9.42 | 5.88 |
| Scribe | 47.03 | 2.38 | 6.59 | 5.44 | 3.52 | 17.29 | 2.99 | 5.46 | 3.27 |
| Voxtral Mini | 63.98 | 6.03 | 10.22 | 8.92 | 6.07 | 12.74 | 5.91 | 7.76 | 4.98 |
| Voxtral Mini Transcribe | 62.01 | 4.71 | 8.25 | 7.29 | 4.85 | 10.42 | 4.37 | 6.70 | 3.96 |
| Voxtral Small | 61.97 | 3.98 | 8.58 | 6.18 | 3.74 | 9.01 | 3.96 | 6.43 | 3.31 |

Table 6: Per‐language WER scores for MLS Dutch, French, German, Italian, Portuguese and Spanish.

| Model | nl | fr | de | it | pt | es |
| --- | --- | --- | --- | --- | --- | --- |
| Whisper large-v3 | 9.19 | 5.09 | 5.72 | 9.78 | 7.03 | 3.89 |
| GPT-4o mini Transcribe | 8.62 | 4.77 | 5.44 | 10.68 | 5.67 | 4.28 |
| Gemini 2.5 Flash | 8.33 | 6.82 | 6.52 | 10.97 | 7.14 | 4.39 |
| Scribe | 38.01 | 5.80 | 9.81 | 12.38 | 15.40 | 8.97 |
| Voxtral Mini | 10.09 | 5.28 | 7.09 | 11.30 | 6.72 | 5.12 |
| Voxtral Mini Transcribe | 9.63 | 4.14 | 5.64 | 9.28 | 5.17 | 3.87 |
| Voxtral Small | 9.43 | 3.73 | 5.57 | 8.44 | 5.85 | 3.62 |

### A.2 Speech Understanding - Full Results

Table 7: Language pair results for the FLEURS speech translation benchmark. Whisper only supports X $\rightarrow$ en translation.

| Model | en $\rightarrow$ de | en $\rightarrow$ es | en $\rightarrow$ fr | en $\rightarrow$ it | de $\rightarrow$ en | es $\rightarrow$ en | fr $\rightarrow$ en | it $\rightarrow$ en |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Whisper large-v3 | \- | \- | \- | \- | 46.1 | 34.9 | 43.0 | 35.7 |
| GPT-4o mini Audio | 44.5 | 36.5 | 52.7 | 37.3 | 51.8 | 41.6 | 48.2 | 41.5 |
| Gemini 2.5 Flash | 44.6 | 36.3 | 53.9 | 37.3 | 39.4 | 32.9 | 42.0 | 31.8 |
| Voxtral Mini | 38.4 | 34.9 | 49.7 | 34.2 | 49.6 | 41.1 | 48.2 | 41.4 |
| Voxtral Small | 47.0 | 39.9 | 57.3 | 39.9 | 56.6 | 46.3 | 54.2 | 46.8 |

Table 8: Per-task accuracy scores for all speech understanding benchmarks. Speech-synthesized subsets of text benchmarks are denoted with <sup>*</sup>.

| Model | Llama QA | Openbook QA | MMLU <sup>*</sup> | MMAU <sup>*</sup> | Trivia QA <sup>*</sup> | GSM8k <sup>*</sup> | AU Bench |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GPT-4o mini Audio | 74.3 | 83.7 | 72.6 | 63.4 | 83.7 | 90.8 | 80.0 |
| Gemini 2.5 Flash | 66.3 | 94.7 | 84.8 | 64.3 | 83.9 | 94.2 | 88.6 |
| Voxtral Mini | 54.3 | 59.6 | 47.6 | 57.1 | 54.9 | 71.6 | 85.6 |
| Voxtral Small | 71.7 | 88.4 | 74.3 | 62.2 | 79.4 | 89.7 | 86.6 |

### A.3 Synthetic Benchmarks

When synthesizing text benchmarks into speech form, a subset of prompts that contain math or code can be deterministically rewritten into speech-compatible text. We refer to this subset as "Verbalizable with Rewrite". The following is the prompt we used with Mistral Large to rewrite the text-prompts:

<svg height="866.43" id="A1.SS3.p2.pic1" overflow="visible" version="1.1" viewBox="0 0 477.38 866.43" width="477.38"><g fill="#000000" stroke="#000000" stroke-width="0.4pt" style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="translate(0,866.43) matrix(1 0 0 -1 0 0)"><g fill="#FFBF80" fill-opacity="1.0" style="--ltx-fill-color:#FFBF80;"><path d="M 0 0 L 0 866.43 L 477.38 866.43 L 477.38 0 Z" style="stroke:none"></path></g><g fill="#FFF2E6" fill-opacity="1.0" style="--ltx-fill-color:#FFF2E6;"><path d="M 1.97 1.97 L 1.97 864.46 L 475.41 864.46 L 475.41 1.97 Z" style="stroke:none"></path></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 21.65 13.78)"><foreignObject color="#000000" height="838.87" overflow="visible" style="--ltx-fg-color:#000000;--fo_width :31.37em;--fo_height:60.62em;--fo_depth :0em;" transform="matrix(1 0 0 -1 0 838.87)" width="434.07"><span style="width:33.91em;"><span style="font-size:90%;">Below is a question datapoint containing a user’s question. I would like to generate a speech version of this question. Therefore, please rewrite my question data according to the following requirements:<br>1. The question should not contain content that cannot be synthesized by a Text To Speech(TTS) model. Numbers should be written in English words rather than Arabic or roman numerals. If they seem to be roman numerals after names of kings and queens, say it as the second, or the third corresponding to the roman number. If the instruction contains only a number, just write it in spoken form.<br>2. The question should be relatively brief without excessive verbiage.<br>3. Expand abbreviations and acronyms (e.g., ’macOS’ as ’mac O S’, ’TensorRT’ as ’Tensor R T’, ’CMake as C Make’, ’JDBC’ as ’Java Database Connectivity’, ’API as A. P. I.’). An abbreviation is hard for a TTS model to say because its not a legitimate english word. Its better to break it up into capital characters.<br>4. If there are number bullets, asterisk bullets, hyphen bullets or dot bullets and the bullets do not seem like options being given by user in the instruction, list them as first, second, lastly or number one, number two and so on. Only if the bullets start with alphabets, use corresponding alphabets like A, B, C, D or use Option A, Option B, Option C, Option D. If bullets start with Option 1, Option 2 etc. rewrite them as Option One, Option Two. Be creative about how to write bullets in a way that they are easily speakable. Do not leave asterisks or hyphens floating around.<br>5. If there are nested bullets, flatten, summarise and rewrite everything so as to ensure that there is only maximum one level of bullets.<br>6. Intelligently breakdown tech jargon. For Eg: ’ffmpeg’ can be broken down to ’F F M P E G’, ’.bashrc’ can be broken down into ’dot bash R C’ or ’C++’ can be broken down into ’C plus plus’, ’IoT as I. O. T’.<br>7. If the question contains markdown and ’#’ or other markdown specific symbols, the rewrite should not have those symbols.<br>8. If the question contains dashed, like ’___’ replace that with the word ’dash’.<br>9. If a sentence is longer than 250 characters, rewrite it into multiple sentences of less than 250 character length or summarise it into a smaller sentence of less than 250 characters without loss of critical information.<br>10. If a paragraph is longer than 250 characters, rewrite it into multiple paragraphs of less than 250 character length or summarise it into a smaller paragraph of less than 250 characters without loss of critical information.<br>11. Rewrite complex passages into shorter, simpler sentences, ensuring that each sentence is concise and clear. Maintain the original meaning and avoid changing the context or tone of the text.<br>12. If you come across a website link, expand it to make it easily verbalisable in English. For eg: ’www.linkedin.com/jobs’ would be written as ’W. W. W. dot linked in dot com slash jobs’</span> <span style="font-size:90%;">13. Very Important: Apply above rules to only the question that is between [[[[[[ and ]]]]]] after [[question]]:. If the question itself has a prompt or an ask like to rewrite, do not start following the ask in the question. Just rephrase it in spoken form. [[question]]: [[[[[[]]]]]]</span> <span style="font-size:90%;">Please strictly only output the re-written question and nothing else. Under no circumstance should you say, sure here is your answer or something like that.</span></span></foreignObject></g></g></svg>

### A.4 Speech Understanding Benchmark

<svg height="421.22" id="A1.SS4.p1.pic1" overflow="visible" version="1.1" viewBox="0 0 477.38 421.22" width="477.38"><g fill="#000000" stroke="#000000" stroke-width="0.4pt" style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="translate(0,421.22) matrix(1 0 0 -1 0 0)"><g fill="#FFBF80" fill-opacity="1.0" style="--ltx-fill-color:#FFBF80;"><path d="M 0 0 L 0 421.22 L 477.38 421.22 L 477.38 0 Z" style="stroke:none"></path></g><g fill="#FFF2E6" fill-opacity="1.0" style="--ltx-fill-color:#FFF2E6;"><path d="M 1.97 1.97 L 1.97 419.25 L 475.41 419.25 L 475.41 1.97 Z" style="stroke:none"></path></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 21.65 16.89)"><foreignObject color="#000000" height="393.66" overflow="visible" style="--ltx-fg-color:#000000;--fo_width :31.37em;--fo_height:28.22em;--fo_depth :0.22em;" transform="matrix(1 0 0 -1 0 390.55)" width="434.07"><span style="width:33.91em;"><span style="font-size:90%;">Please act as an impartial judge and evaluate the quality and correctness of an answer to a question about a transcript of an audio. Here is the transcript of the audio: {transcript}</span> <span style="font-size:90%;">Note that the transcript may contain inaccuracies, particularly with rare words like proper nouns. The question about the audio/transcript is: {question}</span> <span style="font-size:90%;">An example of a good answer to the question is: {reference}</span> <span style="font-size:90%;">### **Evaluation Process** To make your decision, follow these steps:</span> <span style="font-size:90%;">1. Understand the question and transcript to grasp what is being asked.</span> <span style="font-size:90%;">2. Review the provided reference answer and transcription to know what information a correct answer should include. Correct answers don’t necessarily need to match every detail in the reference answer - the reference is just there for you to have an idea on what a good answer looks like.</span> <span style="font-size:90%;">3. Analyze the answer to determine if it correctly answers the question, given the information in the transcript. Also take into consideration the helpfulness and clarity of the answer - it should be presented in a clear, engaging, informative manner.</span> <span style="font-size:90%;">4. After providing your analysis/explanation, provide a score for the answer, {rubric}.</span> <span style="font-size:90%;">### Expected Output Format:</span> <span style="font-size:90%;">Always provide your response in the following JSON format: {{"explanation": "str", "score": bool}}. Don’t output anything other than the JSON object.</span> <span style="font-size:90%;">The answer for you to judge is: {candidate}.</span></span></foreignObject></g></g></svg>

For binary judge, we provide the following rubric:

<svg height="52.81" id="A1.SS4.p3.pic1" overflow="visible" version="1.1" viewBox="0 0 477.38 52.81" width="477.38"><g fill="#000000" stroke="#000000" stroke-width="0.4pt" style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="translate(0,52.81) matrix(1 0 0 -1 0 0)"><g fill="#FFBF80" fill-opacity="1.0" style="--ltx-fill-color:#FFBF80;"><path d="M 0 0 L 0 52.81 L 477.38 52.81 L 477.38 0 Z" style="stroke:none"></path></g><g fill="#FFF2E6" fill-opacity="1.0" style="--ltx-fill-color:#FFF2E6;"><path d="M 1.97 1.97 L 1.97 50.84 L 475.41 50.84 L 475.41 1.97 Z" style="stroke:none"></path></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 21.65 13.78)"><foreignObject color="#000000" height="25.25" overflow="visible" style="--ltx-fg-color:#000000;--fo_width :31.37em;--fo_height:1.82em;--fo_depth :0em;" transform="matrix(1 0 0 -1 0 25.25)" width="434.07"><span style="width:33.91em;"><span style="font-size:90%;">where the score is 1 if the student’s answer is correct and helpful, and 0 otherwise</span></span></foreignObject></g></g></svg>

For grade judge, we provide the following rubric:

<svg height="71.84" id="A1.SS4.p5.pic1" overflow="visible" version="1.1" viewBox="0 0 477.38 71.84" width="477.38"><g fill="#000000" stroke="#000000" stroke-width="0.4pt" style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="translate(0,71.84) matrix(1 0 0 -1 0 0)"><g fill="#FFBF80" fill-opacity="1.0" style="--ltx-fill-color:#FFBF80;"><path d="M 0 0 L 0 71.84 L 477.38 71.84 L 477.38 0 Z" style="stroke:none"></path></g><g fill="#FFF2E6" fill-opacity="1.0" style="--ltx-fill-color:#FFF2E6;"><path d="M 1.97 1.97 L 1.97 69.87 L 475.41 69.87 L 475.41 1.97 Z" style="stroke:none"></path></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 21.65 16.2)"><foreignObject color="#000000" height="44.28" overflow="visible" style="--ltx-fg-color:#000000;--fo_width :31.37em;--fo_height:3.02em;--fo_depth :0.17em;" transform="matrix(1 0 0 -1 0 41.86)" width="434.07"><span style="width:33.91em;"><span style="font-size:90%;">where the score can range from 0 to 5, with 0 meaning the student’s answer is completely wrong and unhelpful, and 5 if the student’s answer is correct and well presented</span></span></foreignObject></g></g></svg>

[^1]: Rosana Ardila, Megan Branson, Kelly Davis, Michael Henretty, Michael Kohler, Josh Meyer, Reuben Morais, Lindsay Saunders, Francis M. Tyers, and Gregor Weber. Common Voice: A Massively-Multilingual Speech Corpus, 2020. URL [https://arxiv.org/abs/1912.06670](https://arxiv.org/abs/1912.06670).

[^2]: Guoguo Chen, Shuzhou Chai, Guanbo Wang, Jiayu Du, Wei-Qiang Zhang, Chao Weng, Dan Su, Daniel Povey, Jan Trmal, Junbo Zhang, Mingjie Jin, Sanjeev Khudanpur, Shinji Watanabe, Shuaijiang Zhao, Wei Zou, Xiangang Li, Xuchen Yao, Yongqing Wang, Yujun Wang, Zhao You, and Zhiyong Yan. GigaSpeech: An Evolving, Multi-domain ASR Corpus with 10,000 Hours of Transcribed Audio. *arXiv e-prints*, art. arXiv:2106.06909, June 2021.

[^3]: Yiming Chen, Xianghu Yue, Chen Zhang, Xiaoxue Gao, Robby T. Tan, and Haizhou Li. VoiceBench: Benchmarking LLM-Based Voice Assistants. *arXiv preprint arXiv:2410.17196*, 2024.

[^4]: Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training Verifiers to Solve Math Word Problems. *arXiv*, October 2021. doi: 10.48550/arXiv.2110.14168.

[^5]: Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. *arXiv preprint arXiv:2507.06261*, 2025.

[^6]: Alexis Conneau, Min Ma, Simran Khanuja, Yu Zhang, Vera Axelrod, Siddharth Dalmia, Jason Riesa, Clara Rivera, and Ankur Bapna. FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech, 2022. URL [https://arxiv.org/abs/2205.12446](https://arxiv.org/abs/2205.12446).

[^7]: Steven B Davis and Paul Mermelstein. Comparison of Parametric Representations for Monosyllabic Word Recognition in Continuously Spoken Sentences. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 28(4):357–366, 1980.

[^8]: Miguel Del Rio, Natalie Delworth, Ryan Westerman, Michelle Huang, Nishchal Bhandari, Joseph Palakapilly, Quinten McNamara, Joshua Dong, Piotr Żelasko, and Miguel Jetté. Earnings-21: A Practical Benchmark for ASR in the Wild. In *Proc. Interspeech 2021*, pages 3465–3469, 2021. doi: 10.21437/Interspeech.2021-1915.

[^9]: Miguel Del Rio, Peter Ha, Quinten McNamara, Corey Miller, and Shipra Chandra. Earnings-22: A Practical Benchmark for Accents in the Wild. *arXiv e-prints*, art. arXiv:2203.15591, March 2022.

[^10]: Qingkai Fang, Shoutao Guo, Yan Zhou, Zhengrui Ma, Shaolei Zhang, and Yang Feng. LLaMA-Omni: Seamless Speech Interaction with Large Language Models. *arXiv*, September 2024. doi: 10.48550/arXiv.2409.06666.

[^11]: J.J. Godfrey, E.C. Holliman, and J. McDaniel. SWITCHBOARD: telephone speech corpus for research and development. In *\[Proceedings\] ICASSP-92: 1992 IEEE International Conference on Acoustics, Speech, and Signal Processing*, volume 1, pages 517–520 vol.1, 1992. doi: 10.1109/ICASSP.1992.225858.

[^12]: Shangmin Guo, Biao Zhang, Tianlin Liu, Tianqi Liu, Misha Khalman, Felipe Llinares, Alexandre Rame, Thomas Mesnard, Yao Zhao, Bilal Piot, Johan Ferret, and Mathieu Blondel. Direct language model alignment from online ai feedback, 2024. URL [https://arxiv.org/abs/2402.04792](https://arxiv.org/abs/2402.04792).

[^13]: Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring Massive Multitask Language Understanding. *arXiv*, September 2020. doi: 10.48550/arXiv.2009.03300.

[^14]: Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. *arXiv preprint arXiv:2410.21276*, 2024.

[^15]: Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke Zettlemoyer. TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension. *ACL Anthology*, pages 1601–1611, July 2017. doi: 10.18653/v1/P17-1147.

[^16]: Mistral-AI, Abhinav Rastogi, Albert Q. Jiang, Andy Lo, Gabrielle Berrada, Guillaume Lample, Jason Rute, Joep Barmentlo, Karmesh Yadav, Kartik Khandelwal, Khyathi Raghavi Chandu, Léonard Blier, Lucile Saulnier, Matthieu Dinot, Maxime Darrin, Neha Gupta, Roman Soletskyi, Sagar Vaze, Teven Le Scao, Yihan Wang, Adam Yang, Alexander H. Liu, Alexandre Sablayrolles, Amélie Héliou, Amélie Martin, Andy Ehrenberg, Anmol Agarwal, Antoine Roux, Arthur Darcet, Arthur Mensch, Baptiste Bout, Baptiste Rozière, Baudouin De Monicault, Chris Bamford, Christian Wallenwein, Christophe Renaudin, Clémence Lanfranchi, Darius Dabert, Devon Mizelle, Diego de las Casas, Elliot Chane-Sane, Emilien Fugier, Emma Bou Hanna, Gauthier Delerce, Gauthier Guinet, Georgii Novikov, Guillaume Martin, Himanshu Jaju, Jan Ludziejewski, Jean-Hadrien Chabran, Jean-Malo Delignon, Joachim Studnia, Jonas Amar, Josselin Somerville Roberts, Julien Denize, Karan Saxena, Kush Jain, Lingxiao Zhao, Louis Martin, Luyu Gao, Lélio Renard Lavaud, Marie Pellat, Mathilde Guillaumin, Mathis Felardos, Maximilian Augustin, Mickaël Seznec, Nikhil Raghuraman, Olivier Duchenne, Patricia Wang, Patrick von Platen, Patryk Saffer, Paul Jacob, Paul Wambergue, Paula Kurylowicz, Pavankumar Reddy Muddireddy, Philomène Chagniot, Pierre Stock, Pravesh Agrawal, Romain Sauvestre, Rémi Delacourt, Sanchit Gandhi, Sandeep Subramanian, Shashwat Dalal, Siddharth Gandhi, Soham Ghosh, Srijan Mishra, Sumukh Aithal, Szymon Antoniak, Thibault Schueller, Thibaut Lavril, Thomas Robert, Thomas Wang, Timothée Lacroix, Valeriia Nemychnikova, Victor Paltz, Virgile Richard, Wen-Ding Li, William Marshall, Xuanyu Zhang, and Yunhao Tang. Magistral, 2025. URL [https://arxiv.org/abs/2506.10910](https://arxiv.org/abs/2506.10910).

[^17]: Mistral AI Team. Un Ministral, des Ministraux, October 2024. URL [https://mistral.ai/news/ministraux](https://mistral.ai/news/ministraux). Accessed: 2025-07-11.

[^18]: Mistral AI Team. Mistral Small 3.1, March 2025. URL [https://mistral.ai/news/mistral-small-3-1](https://mistral.ai/news/mistral-small-3-1). Accessed: 2025-07-11.

[^19]: Eliya Nachmani, Alon Levkovitch, Roy Hirsch, Julian Salazar, Chulayuth Asawaroengchai, Soroosh Mariooryad, Ehud Rivlin, RJ Skerry-Ryan, and Michelle Tadmor Ramanovich. Spoken Question Answering and Speech Continuation Using Spectrogram-Powered LLM, 2024. URL [https://arxiv.org/abs/2305.15255](https://arxiv.org/abs/2305.15255).

[^20]: Tu Anh Nguyen, Benjamin Muller, Bokai Yu, Marta R Costa-Jussa, Maha Elbayad, Sravya Popuri, Christophe Ropers, Paul-Ambroise Duquenne, Robin Algayres, Ruslan Mavlyutov, et al. Spirit-lm: Interleaved spoken and written language model. *Transactions of the Association for Computational Linguistics*, 13:30–52, 2025.

[^21]: Patrick K. O’Neill, Vitaly Lavrukhin, Somshubra Majumdar, Vahid Noroozi, Yuekai Zhang, Oleksii Kuchaiev, Jagadeesh Balam, Yuliya Dovzhenko, Keenan Freyberg, Michael D. Shulman, Boris Ginsburg, Shinji Watanabe, and Georg Kucsko. SPGISpeech: 5,000 Hours of Transcribed Financial Audio for Fully Formatted End-to-End Speech Recognition. In *Proc. Interspeech 2021*, pages 1434–1438, 2021. doi: 10.21437/Interspeech.2021-1860.

[^22]: Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur. Librispeech: An ASR corpus based on public domain audio books. In *2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, pages 5206–5210, 2015. doi: 10.1109/ICASSP.2015.7178964.

[^23]: Vineel Pratap, Qiantong Xu, Anuroop Sriram, Gabriel Synnaeve, and Ronan Collobert. MLS: A Large-Scale Multilingual Dataset for Speech Research. *arXiv preprint arXiv:2012.03411*, 2020.

[^24]: Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust Speech Recognition via Large-Scale Weak Supervision. In *International Conference on Machine Learning*, pages 28492–28518. PMLR, 2023.

[^25]: Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model, 2024. URL [https://arxiv.org/abs/2305.18290](https://arxiv.org/abs/2305.18290).

[^26]: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention Is All You Need. *Advances in Neural Information Processing Systems*, 30, 2017.

[^27]: Emmanuel Vincent, Shinji Watanabe, Aditya Arie Nugraha, Jon Barker, and Ricard Marxer. An Analysis of Environment, Microphone and Data Simulation Mismatches in Robust Speech Recognition. *Comput. Speech Lang.*, 46(C):535–557, nov 2017. ISSN 0885-2308. doi: 10.1016/j.csl.2016.11.005. URL [https://doi.org/10.1016/j.csl.2016.11.005](https://doi.org/10.1016/j.csl.2016.11.005).

[^28]: Changhan Wang, Morgane Riviere, Ann Lee, Anne Wu, Chaitanya Talnikar, Daniel Haziza, Mary Williamson, Juan Pino, and Emmanuel Dupoux. VoxPopuli: A Large-Scale Multilingual Speech Corpus for Representation Learning, Semi-Supervised Learning and Interpretation. In *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)*, pages 993–1003, Online, August 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.acl-long.80. URL [https://aclanthology.org/2021.acl-long.80](https://aclanthology.org/2021.acl-long.80).

[^29]: Aohan Zeng, Zhengxiao Du, Mingdao Liu, Lei Zhang, Shengmin Jiang, Yuxiao Dong, and Jie Tang. Scaling speech-text pre-training with synthetic interleaved data. *arXiv preprint arXiv:2411.17607*, 2024.

[^30]: Yu Zhang, Wei Han, James Qin, Yongqiang Wang, Ankur Bapna, Zhehuai Chen, Nanxin Chen, Bo Li, Vera Axelrod, Gary Wang, Zhong Meng, Ke Hu, Andrew Rosenberg, Rohit Prabhavalkar, Daniel S. Park, Parisa Haghani, Jason Riesa, Ginger Perng, Hagen Soltau, Trevor Strohman, Bhuvana Ramabhadran, Tara Sainath, Pedro Moreno, Chung-Cheng Chiu, Johan Schalkwyk, Françoise Beaufays, and Yonghui Wu. Google usm: Scaling automatic speech recognition beyond 100 languages, 2023. URL [https://arxiv.org/abs/2303.01037](https://arxiv.org/abs/2303.01037).