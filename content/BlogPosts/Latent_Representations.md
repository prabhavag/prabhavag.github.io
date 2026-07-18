---
title: "Understanding Latent Representations"
date: 2025-09-26
tags:
  - latent-representations
  - vae
  - quantization
  - machine-learning
  - deep-learning
draft: false
---
Modern machine learning models—from autoencoders to large language models—rely on the idea of latent representations. These are compact, abstract encodings of data that capture structure, meaning, or features without directly mirroring the raw input. A latent representation is a vector (or set of vectors) in a hidden space learned from data, which encodes the input in a way that makes downstream tasks easier. Latent spaces often have meaningful
geometric properties like clustering of similar items together, meaningful semantic linearity (in case of word embeddings `king - man + woman ≈ queen`)), or the manifold hypothesis which suggests that data lies on a much lower dimensional manifold embedded in latent space. 


## Auto-encoder
Let's assume that our data is represented by $\boldsymbol{x}$, and latent variable is denoted as $\boldsymbol{z}$.
A standard autoencoder (AE) consists of two modules **Encoder:** network, $$z = f_{\theta}(x)$$ and a **Decoder:** 
network $$\hat{x} = g_{\phi}(z)$$. 

**Loss:** 
$$L(x, \hat{x}) = \|x - g_{\phi}(f_{\theta}(x))\|_2$$


<div style="text-align: center;">
  <img src="assets/autoencoder.png" width="50%" alt="Autoencoder Architecture">
  <figcaption><strong>Figure 1:</strong> Autoencoder Architecture</figcaption>
</div>

