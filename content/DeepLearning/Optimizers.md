---
title: "Deep Learning Optimizers"
date: 2026-01-14
tags:
  - optimizers
  - adam
  - adagrad
  - rmsprop
  - muon
  - machine-learning
  - deep-learning
draft: false
---
Once the gradient is calculated during backpropagation, the optimizer uses stochastic gradient descent to update the model's weights. Various augmentations to SGD exist, each adjusting the weights in different ways based on the computed gradients. Here are some of the most commonly used optimizers in deep learning:

## Vanilla Stochastic Gradient Descent (SGD)
$$w_{t+1} = w_t - \eta \nabla L(w_t)$$

## Momentum
Momentum is an extension of SGD that helps accelerate gradient vectors in the right direction, leading to faster convergence. It does this by accumulating a velocity vector in directions of persistent reduction in the loss:

$$v_{t+1} = \gamma v_t + \eta \nabla L(w_t)$$
$$w_{t+1} = w_t - v_{t+1}$$

where $\gamma$ is the momentum coefficient (typically 0.9).



