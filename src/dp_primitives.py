"""
dp_primitives.py
Minimal, verifiable drop-in replacements for the two diffprivlib primitives
dp_cohorts.py depends on (DPKMeans, Laplace). Implemented directly so the
pipeline can run in network-restricted environments without changing any
call site in dp_cohorts.py.

Both mechanisms are standard, well-documented DP building blocks:

  Laplace mechanism (Dwork & Roth, 2014):
      Given a numeric query f(D) with L1-sensitivity Δ, releasing
      f(D) + Lap(Δ/ε) satisfies ε-differential privacy.

  DP k-means (via noisy Lloyd's algorithm, Su et al. 2016 style):
      Standard k-means, but at each iteration the per-cluster sums used
      to compute new centroids are perturbed with Laplace noise before
      being divided by (noised) counts. Splits the epsilon budget evenly
      across iterations.
"""

from __future__ import annotations
import numpy as np


class Laplace:
    """
    Matches diffprivlib.mechanisms.Laplace's interface: Laplace(epsilon=..., sensitivity=...).randomise(value)
    """
    def __init__(self, epsilon: float, sensitivity: float):
        if epsilon <= 0:
            raise ValueError("epsilon must be > 0")
        if sensitivity < 0:
            raise ValueError("sensitivity must be >= 0")
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        # scale = sensitivity / epsilon (standard Laplace mechanism)
        self.scale = sensitivity / epsilon if epsilon > 0 else 0.0

    def randomise(self, value: float) -> float:
        if self.scale == 0:
            return float(value)
        noise = np.random.laplace(loc=0.0, scale=self.scale)
        return float(value) + float(noise)


class DPKMeans:
    """
    Matches diffprivlib.models.KMeans's interface used in dp_cohorts.py:
        DPKMeans(n_clusters=k, epsilon=eps, bounds=(lower_array, upper_array))
        .fit_predict(X) -> cluster labels

    Implementation: noisy Lloyd's algorithm.
      - Data is assumed already scaled into `bounds` (dp_cohorts.py pre-scales to [0,1]^d
        with PUBLIC bounds before calling this, matching diffprivlib's own convention).
      - Budget epsilon is split evenly across `n_iter` Lloyd iterations.
      - Each iteration: assign points to nearest centroid (non-private — centroids
        are already DP from the previous round, and assignment doesn't touch raw
        data further under standard treatments of this mechanism family), then
        release noisy per-cluster (sum, count) via the Laplace mechanism and
        recompute centroids from the noisy quantities.
      - Centroids initialised via a fixed, data-independent grid over `bounds`
        (no privacy cost — does not depend on the data).
    """

    def __init__(self, n_clusters: int, epsilon: float, bounds: tuple, n_iter: int = 6, random_state: int = 42):
        self.n_clusters = n_clusters
        self.epsilon = epsilon
        self.bounds = bounds  # (lower_array, upper_array)
        self.n_iter = n_iter
        self.random_state = random_state
        self.cluster_centers_: np.ndarray | None = None

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        rng = np.random.RandomState(self.random_state)
        lower, upper = self.bounds
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        d = X.shape[1]
        n = X.shape[0]
        k = self.n_clusters

        # Data-independent initialisation: deterministic jittered grid over the public bounds.
        # (No privacy cost: does not look at X.)
        centers = lower + (upper - lower) * rng.uniform(0.15, 0.85, size=(k, d))

        eps_per_iter = self.epsilon / max(self.n_iter, 1)
        # Sensitivity bounds for per-cluster sum (L1 over d dims) and count.
        range_l1 = float(np.sum(upper - lower))

        labels = np.zeros(n, dtype=int)
        for it in range(self.n_iter):
            # Assign each point to nearest current centroid.
            dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
            labels = np.argmin(dists, axis=1)

            new_centers = centers.copy()
            for c in range(k):
                mask = labels == c
                count_true = int(mask.sum())

                # Noisy count (sensitivity 1 per iteration's count query).
                count_noisy = Laplace(epsilon=eps_per_iter / 2, sensitivity=1.0).randomise(count_true)
                count_noisy = max(count_noisy, 1.0)  # avoid division blow-ups

                if count_true > 0:
                    true_sum = X[mask].sum(axis=0)
                else:
                    true_sum = np.zeros(d)

                # Noisy sum, one Laplace draw per dimension, sensitivity = per-dim range.
                noisy_sum = np.array([
                    Laplace(epsilon=eps_per_iter / (2 * d), sensitivity=(upper[j] - lower[j])).randomise(true_sum[j])
                    for j in range(d)
                ])

                candidate = noisy_sum / count_noisy
                new_centers[c] = np.clip(candidate, lower, upper)

            centers = new_centers

        self.cluster_centers_ = centers
        # Final assignment with the last (DP) centroids.
        dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        return labels