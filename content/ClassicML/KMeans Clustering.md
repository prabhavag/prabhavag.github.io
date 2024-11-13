---
tags: kmeans, clustering, unsupervised-learning
---
## Problem
Given a set of $n$ points $(x_1, x_2, ..., x_n)$ where $x_i \in R^d$, assign the points to $k \leq n$ clusters  such that the following loss function should be minimized. [@wikipedia_kmeans]. Formally, 

$$
L(\mu) = \sum\limits_{i=1}^{n}\min\limits_{j \in {1,...,k}}\|x_i - c_j\|_2^2
$$
## Algorithm (Naive K-Means)
The problem is [NP-hard](https://en.wikipedia.org/wiki/NP-hardness) wrt finding the centroids which minimize the WCSS, but is approximated in practice using the following iterative refinement technique

- Initialize the $k$ centroids $(\mu_1, \mu_2, ..., \mu_k)$ where $\mu_j \in R^d$ randomly
- Repeat until convergence or max iterations $I$
    - *Assignment step*: For every point $x_i$, assign cluster label $l_i$ as the cluster with minimum $L^2$ distance:
        $$
        l_{i} = \arg\min_{j}\|x_{i} - \mu_j\|_2^2 
         $$
    - *Update step: Update centroid $c_j$ as:
    $$
    \mu_j = \frac{1}{|\{i : l_i = j\}|}\sum\limits_{i: l_i = j}x_i
    $$
- Time complexity for the above algorithm is $O(Inkd)$ 

### Proof of Convergence
**Lemma 1**: The loss function is monotonically non-increasing for both the assignment and update step. [@krause2016]
**Proof**: Let $l = (l_1, l_2, ..., l_n)$ denotes the cluster assignment for the $n$ points.

*Assignment step*: The loss function $L(\mu, l)$ can be written as:
$$
    L(\mu, l) = \sum\limits_{i=1}^{n} \|x_i - \mu_{l_i} \|_2^2
$$
Consider a data point $x_i$ , and let $l_i$ be the assignment from the previous iteration and $z^∗_i$ be the new assignment obtained as:
$$
l^∗_i \in \arg \min_{j \in {1,...,k}} \|x_i − \mu_j\|_2^2
$$
Let $l^*$ denote the new cluster assignment, the change in loss function can be written as 
$$
L(\mu, l^*) - L(\mu, l) = \sum\limits_{i=1}^{n} (\|x_i - \mu_{l^{*}_{i}} \|_2^2 - \|x_i - \mu_{l_i}\|_2^2) \leq 0
$$
The above inequality holds because $l^*_i$ assigns each $x_i$ to the nearest cluster.

*Update step*: The loss function $L(\mu, l)$ can be alternatively written as:
$$
L(\mu, l) = \sum\limits_{j=1}^{k}(\sum\limits_{i:l_i=j}\|x_i - \mu_j\|_2^2)
$$
For the $j^{th}$ cluster, let's denote the previous centroid as $\mu_j$ and the updated one as $\mu_j^*$ 
$$
\mu_j^* = \frac{1}{|\{i : l_i = j\}|}\sum\limits_{i: l_i = j}x_i  
$$
Let $\mu^*$ denote the new centroids for all the $k$ clusters, change in loss function is given as:
$$
L(\mu^*, l) - L(\mu, l) = \sum\limits_{j=1}^{k}((\sum\limits_{i:l_i=j}\|x_i - \mu_j^*\|_2^2) - (\sum\limits_{i:l_i=j}\|x_i - \mu_j\|_2^2) \leq 0
$$
We can verify after taking gradients of $L(\mu, l)$ wrt each $\mu_j$ that $L(\mu, l)$ is minimized with the updated definition of $\mu_j^*$, that's why the inequality holds.

**Claim**: The K-Means algorithm terminate in finite number of steps.
1.  If the clustering assignment changes, the newer one will have a lower cost (from Lemma 1)
2. If the cluster assignment doesn't change, the centroids don't change, and the algorithm terminates.
Since the number of clusterings is finite and equal to $k^n$, the algorithm will eventually hit condition 2, and then terminate.
## Implementation (in python using pytorch)

```python
import torch
import torch.nn.functional as F
from typing import Tuple

# Two-loop solution
def k_means_clustering_twoloops(
    data: torch.Tensor,
    centers: torch.Tensor,
    max_iterations: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
        data: Points to be clustered [N, D]
        centers: Centroid matrix [K, D]
        max_iterations: max iterations clustering should run
    """
    for i in range(max_iterations):
        dist = torch.zeros((data.shape[0], centers.shape[0]))  # (N, K)
        for j in range(data.shape[0]):
            dist_j = torch.sqrt(
                ((data[j : j + 1] - centers) ** 2).sum(axis=1)
            )  # (K, )
            dist[j, :] = dist_j

        labels = torch.argmin(dist, dim=1)  # (N,)
        for j in range(centers.shape[0]):
            centers[j] = torch.mean(data[labels == j, :], dim=0)

    return centers, labels


# One-loop solution
def k_means_clustering_oneloop(
    data: torch.Tensor,
    centers: torch.Tensor,
    max_iterations: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
        data: Points to be clustered [N, D]
        centers: Centroid matrix [K, D]
        max_iterations: max iterations clustering should run
    """
    N, _ = data.shape
    for i in range(max_iterations):
        dist = torch.sqrt(
            torch.sum(data**2, dim=1).unsqueeze(1)
            + torch.sum(centers**2, dim=1).unsqueeze(0)
            - 2 * data @ centers.T
        )  # [N, K]
        labels = torch.argmin(dist, dim=1)  # [N]

        # create [K, N] matrix M which has M(i, j) = 1 if point j belongs to
        # cluster i, zero otherwise
        centers_to_labels = torch.zeros_like(dist).T  # [K, N]
        centers_to_labels[labels, torch.arange(N)] = 1
        centers_to_labels = centers_to_labels / torch.max(
            torch.sum(centers_to_labels, dim=1), torch.tensor(1e-12)
        ).unsqueeze(1)
        centers = centers_to_labels @ data  # [K, D]
    return centers, labels
```


![[kmeans_before.png]]


![[kmeans_after.png]]

### Run-time comparison

```python
import timeit

# Time the two-loop k-means function
time_two_loops = timeit.timeit(
    lambda: k_means_clustering_twoloops(data, centers, max_iter), number=10
)
print(f"Time taken for two-loop k-means: {time_two_loops:.6f} seconds")

# Time the one-loop k-means function
time_one_loop = timeit.timeit(
    lambda: k_means_clustering_oneloop(data, centers, max_iter), number=10
)
print(f"Time taken for one-loop k-means: {time_one_loop:.6f} seconds")
```

Time taken for two-loop k-means: 2.066990 seconds
Time taken for one-loop k-means: 0.246190 seconds

Single time version is $\sim 8.5$ faster than the two-loops version for a small datasets of 100 2-dim points with 10 iterations

Check out the [colab notebook](https://colab.research.google.com/drive/1EKSTa5acLJaR3KMo2CRNZgDWK98W0eTp#scrollTo=wL_sS6g1HK93) for more details.
## Further Improvements

### Improved Initialization of Centroids

KMeans++

### How to select the value of K

## References

