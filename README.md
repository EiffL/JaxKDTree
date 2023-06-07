# JaxKDTree

A Jax wrapper for [cudaKDTree](https://github.com/ingowald/cudaKDTree), a Library for Building and Querying Left-Balanced (point-)kd-Trees in CUDA. See [Wald 2022](https://arxiv.org/abs/2211.00120) for details of this algorithm.

**This is still very experimental code, contributions and improvements most welcome.**

```python
import jax.numpy as jnp
import jaxkdtree

# Define an array of positions of shape [N, 3]
pos = jax.random.normal(jax.random.PRNGKey(0), (64**3, 3))

# Compute the indices of the nearest neighbors
nn_inds = jaxkdtree.kNN(pos, k=8, max_radius=1.0)
```

On A100, as an indication, it gives you k=8 nearest neighboors for:
  - 64^3 particles : 14 ms
  - 128^3 particles:  90 ms
  - 256^3 particles: 600 ms

Checkout the demo notebook: <a href="https://colab.research.google.com/github/EiffL/JaxKDTree/blob/main/notebooks/knn_demo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Install

This assumes that you have jax, CUDA, and cmake installed on your system:
```bash
$ pip install git+https://github.com/EiffL/JaxKDTree.git
```
