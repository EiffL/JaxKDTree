# JaxKDTree
A Jax wrapper for [cudaKDTree](https://github.com/ingowald/cudaKDTree)

On A100, as an indication, it gives you k=8 nearest neighboors for:
  - 64^3 particles : 14 ms
  - 128^3 particles:  90 ms
  - 256^3 particles: 600 ms

Checkout the demo notebook: https://github.com/EiffL/JaxKDTree/blob/main/notebooks/knn_demo.ipynb
