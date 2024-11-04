## Problem
Given a set of 


## Two-loop Solution
```python
# K-Means Clustering
# Write a Python function that implements the k-Means algorithm for clustering,
# starting with specified initial centroids and a set number of iterations.
# The function should take a list of points (each represented as a tuple of coordinates),
# an integer k representing the number of clusters to form, a list of initial centroids (each a tuple of coordinates),
# and an integer representing the maximum number of iterations to perform. The function will iteratively assign each point
# to the nearest centroid and update the centroids based on the assignments until the centroids do not change significantly,
# or the maximum number of iterations is reached. The function should return a list of the final centroids of the clusters.
# Round to the nearest fourth decimal.

# Two for loop solution
# Time: O(N*K*D*I)
import numpy as np


def k_means_clustering(
    points: list[tuple[float, float]],
    k: int,
    initial_centroids: list[tuple[float, float]],
    max_iterations: int,
) -> list[tuple[float, float]]:
    # Need to convert them to float, otherwise they can remain
    # as integers and dynamically casted to int somewhere
    points = np.array(points, dtype=np.float32)  # (N, 2)
    centroids = np.array(initial_centroids, dtype=np.float32)  # (K, 2)

    for i in range(max_iterations):
        dist = np.zeros((points.shape[0], centroids.shape[0]))  # (N, K)
        for j in range(points.shape[0]):
            dist_j = np.sqrt(
                ((points[j : j + 1] - centroids) ** 2).sum(axis=1)
            )  # (K, )
            dist[j, :] = dist_j

        labels = np.argmin(dist, axis=1)  # (N,)
        for j in range(centroids.shape[0]):
            centroids[j] = np.mean(points[labels == j, :], axis=0)

    final_centroids = [tuple(centroids[i]) for i in range(centroids.shape[0])]
    return final_centroids


if __name__ == "__main__":
    print(
        k_means_clustering(
            [(0, 0, 0), (2, 2, 2), (1, 1, 1), (9, 10, 9), (10, 11, 10), (12, 11, 12)],
            2,
            [(1, 1, 1), (10, 10, 10)],
            10,
        )
    )
```

## One-loop Solution
```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)


# # Bonus: Implement the k-Means algorithm using a single loop instead of nested loops.
# # Return the final centroids of the clusters.
def k_means_clustering_oneloop(
    data: torch.Tensor,
    centers: torch.Tensor,
    max_iterations: int,
) -> torch.Tensor:
    pass
    """
        data: Points to be clustered [N, D]
        centers: Centroid matrix [K, D]
        max_iterations: max iterations clustering should run
    """
    N, _ = data.shape()
    for i in range(max_iterations):
        dist = torch.sqrt(
            torch.sum(data**2, dim=1).unsqueeze(1)
            + torch.sum(centers**2, dim=1).unsqueeze(0)
            - 2 * data @ centers.T
        )  # [N, K]
        labels = torch.argmin(dim=1)  # [N]

        # create [K, N] matrix M which has M(i, j) = 1 if point j belongs to
        # cluster i, zero otherwise
        centers_to_labels = torch.zeros_like(dist).T  # [K, N]
        centers_to_labels[labels, torch.arange(N)] = 1
        centers_to_labels = centers_to_labels / torch.max(
            torch.sum(centers_to_labels, dim=1), 1e-12
        ).unsqueeze(1)
        centers = centers_to_labels @ data  # [K, D]

```