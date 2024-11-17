---
tags: kmeans, clustering, unsupervised-learning
---
## Problem
Given a set of $n$ points $X = (x_1, x_2, ..., x_n)$ where $x_i \in R^d$, assign the points to $k \leq n$ clusters  such that the following loss function should be minimized. [@wikipedia_kmeans]. Formally, 

$$
L(\mu) = \sum\limits_{i=1}^{n}\min\limits_{j \in {1,...,k}}\|x_i - \mu_j\|_2^2
$$
## Algorithm (Naive KMeans)
The problem is [NP-hard](https://en.wikipedia.org/wiki/NP-hardness) wrt finding the centroids which minimize the WCSS, but is approximated in practice using the following iterative refinement technique in naive KMeans:

1. Initialize the $k$ centroids $(\mu_1, \mu_2, ..., \mu_k)$ where $\mu_j \in R^d$ randomly
2. Repeat until convergence or max iterations $I$
    - *Assignment step*: For every point $x_i$, assign cluster label $l_i$ as the cluster with minimum $L^2$ distance:
    $$
    l_{i} = \arg\min_{j}\|x_{i} - \mu_j\|_2^2 
    $$
    - *Update step*: Update centroid $c_j$ as:
    $$
    \mu_j = \frac{1}{|\{i : l_i = j\}|}\sum\limits_{i: l_i = j}x_i
    $$

    - *Empty Cluster Handling*: Check for clusters with zero assigned points and re-initialize them randomly using data points
Time complexity for the above algorithm is $O(Inkd)$ 

### Proof of Convergence
**Lemma 1**: The loss function is monotonically non-increasing for both the assignment and update step. [@krause2016lis]
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
                ((data[j : j + 1] - centers) ** 2).sum(dim=1)
            )  # (K, )
            dist_j_sq = ((data[j : j + 1] - centers) ** 2).sum(dim=1) # (K, )
            dist[j, :] = dist_j

        labels = torch.argmin(dist, dim=1)  # (N,)
        for j in range(centers.shape[0]):
            points_in_cluster = data[labels == j, :]
            if points_in_cluster.shape[0] > 0:
                centers[j] = torch.mean(points_in_cluster, dim=0)
            else:
                # No point assigned to this cluster, pick a data point arbitraily as center
                centers[j] = data[torch.randint(0, data.shape[0], (1,)), :]

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
        # Alternatively torch.cdist API can be used
        dist = torch.sqrt(
            torch.sum(data**2, dim=1).unsqueeze(1)
            + torch.sum(centers**2, dim=1).unsqueeze(0)
            - 2 * data @ centers.T
        )  # [N, K]
        labels = torch.argmin(dist, dim=1)  # [N]

        # create [K, N] matrix M which has M(i, j) = 1 if point j belongs to cluster i, zero otherwise
        centers_to_labels = torch.zeros_like(dist).T  # [K, N]
        centers_to_labels[labels, torch.arange(N)] = 1
        centers_to_labels = centers_to_labels / torch.max(
            torch.sum(centers_to_labels, dim=1), torch.tensor(1e-12)
        ).unsqueeze(1)
        centers = centers_to_labels @ data  # [K, D]

        # Reinitialize the centers which didn't get a point based on random rows from data
        # Identify rows in centers that are all zeros
        zero_rows_mask = (centers == 0.0).all(dim=1)  # Boolean mask for zero rows in centers

        # Count how many rows are zero
        num_zero_rows = zero_rows_mask.sum().item()
        random_indices = torch.randint(0, data.shape[0], (num_zero_rows,))
        random_rows_from_data = data[random_indices]  # Select random rows

        # Replace zero rows in centers with the selected rows from data
        centers[zero_rows_mask] = random_rows_from_data
    return centers, labels

```

### Clustering Example
<div style="text-align: center;" >
  <img src="assets/kmeans_before.png" width="50%">
  <figcaption> Fig 1. Data Points
  </figcaption>
</div>



<div style="text-align: center;" >
  <img src="assets/kmeans_after.png" width="50%">
  <figcaption>Fig 2. KMeans Clustering - One Loop After 100 Iterations
  </figcaption>
</div>


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

Time taken for two-loop k-means: 41.656580 seconds 
Time taken for one-loop k-means: 0.507115 seconds

Single time version is $\sim 85$ faster than the two-loops version for a small datasets of 100 2-dim points with 10 iterations
## Improved Initialization of Centroids with KMeans++

Naive KMeans initializes the centroids randomly, which could lead to convergence to bad local optimum. Here is an example below:

<div style="text-align: center;" >
  <img src="assets/kmeans_poor_example.png" width="50%">
  <figcaption>Poor clustering example due to bad initialization of centroids
  </figcaption>
</div>

K-Means++ aims to solve the initialization problem, while providing $\Theta(log k)$-competitive accuracy guarantees. The intuition is to select the initial centers which are further apart from each other. The authors provide preliminary resulting demonstrating that KMeans++ leads to both improvements in speed and accuracy in practice. [@arthur2007kmeans].  The algorithm works as follows:

1. Choose an initial center $\mu_1$ uniformly at random from $X$. 
2. Choose the next center $\mu_i$ , selecting $\mu_i = x' \in X$ with probability $\frac{D(x')^2} {\sum\limits_{x \in X} D(x)^2}$, where  $D(x)$ denote the shortest distance from a data point $x'$ to the closest center we have already chosen.
3. Repeat Step 2. until we have chosen a total of k centers. 
4. Proceed as with the standard k-means algorithm.

```python
import torch

def kmeans_pp_initialization(data: torch.Tensor, num_clusters: int) -> torch.Tensor:
    """
    Performs k-means++ initialization.

    Args:
        data: The data points to be clustered [N, D].
        num_clusters: The desired number of clusters (K).

    Returns:
        A tensor of initial cluster centers [K, D].
    """
    n_samples = data.shape[0]
    centers = torch.zeros(num_clusters, data.shape[1], dtype=data.dtype)
    # Choose the first center randomly from the data points
    centers[0] = data[torch.randint(0, n_samples, (1,))]

    for i in range(1, num_clusters):
      distances = torch.min(torch.cdist(data, centers[:i]), dim=1)[0] # [N]
      probabilities = distances / torch.sum(distances)
      cumulative_probabilities = torch.cumsum(probabilities, dim=0) # [N]

      # Generate random number
      rand_val = torch.rand(1)

      # Find the index of the next center based on cumulative probabilities
      next_center_index = torch.searchsorted(cumulative_probabilities, rand_val)
      centers[i] = data[next_center_index]

    return centers
```

Check out the [colab notebook](https://colab.research.google.com/drive/1EKSTa5acLJaR3KMo2CRNZgDWK98W0eTp#scrollTo=wL_sS6g1HK93) for all the code pointers and plots.
## How to select the number of clusters
There are several heuristics to identify the number of clusters $K$ based on Elbow Method, Silhouette Score, cross-validation or Dunn index, either using the KMeans loss $L(\mu)$ or other good of fitness metrics. [@neptune_kmeans]

##  Further Readings
[Dasgupta Kmeans Handout](https://cseweb.ucsd.edu/~dasgupta/291-geom/kmeans.pdf)
[GeeksForGeeks KMeans](https://www.geeksforgeeks.org/ml-k-means-algorithm/)
[Elkan Kmeans - Speed up using triangle inequality](https://cdn.aaai.org/ICML/2003/ICML03-022.pdf)
[Stanford CS221 KMeans Handout - KMeans Compared with EM Algorithm](https://stanford.edu/~cpiech/cs221/handouts/kmeans.html)
## References