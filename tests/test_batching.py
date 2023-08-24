import time
import jax
import jax.numpy as np
import jaxkdtree

n_nodes = 100000
n_batches = [1, 4, 64]
x = jax.random.normal(jax.random.PRNGKey(34), (n_batches[-1], n_nodes, 3))

print(f"===================================")
for i in range(10):
    print("No batch dimension")
    start = time.time()
    print(f"Point cloud size: {x[i].shape}")
    print(f"Point cloud sum: {x[i].sum()}")
    res = jaxkdtree.kNN(x[i], 8, 100.0)
    print(f"Time: {time.time() - start} s")
    print(f"Output size: {res.shape}")
    print(f"Sum of first: {res.sum()}")
    print(f"===================================")

# Test batching 

for n_batch in n_batches:
    print("With batch dimension")
    start = time.time()  
    print(f"Point cloud size: {x[:n_batch].shape}")
    print(f"Point cloud sum: {x[:10].sum((-1, -2))}")
    res = jax.vmap(jaxkdtree.kNN, in_axes=(0,None,None))(x[:n_batch], 8, 100.0)
    # res = jaxkdtree.kNN(x[:n_batch], 8, 100.0)  # The bare op can handle batched inputs, so vmap isn't strictly necessary
    print(f"Time: {time.time() - start} s")
    print(f"Output size: {res.shape}")
    print(f"Sum of first 10: {res[:10].sum((-1, -2))}")
    print(f"===================================")

# Test jit

@jax.jit
def get_knn(x):
    return jaxkdtree.kNN(x, 8, 100.0)

for n_batch in n_batches:
    print("With batch dimension")
    start = time.time()  
    print(f"Point cloud size: {x[:n_batch].shape}")
    print(f"Point cloud sum: {x[:10].sum((-1, -2))}")
    res = jax.vmap(get_knn)(x[:n_batch])
    print(f"Time: {time.time() - start} s")
    print(f"Output size: {res.shape}")
    print(f"Sum of first 10: {res[:10].sum((-1, -2))}")
    print(f"===================================")
