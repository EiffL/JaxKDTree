import jax
import jax.numpy as np
import jaxkdtree
import pytest

@pytest.fixture
def setup_data():
    """ Generate a random point cloud
    """
    n_nodes = 1000
    n_batch = 32
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (n_batch, n_nodes, 3))
    return x,

def pairwise_distances(point_cloud):
    """ Compute pairwise distances between points in a point cloud.
    """
    dr = point_cloud[:, None, :] - point_cloud[None, :, :]
    distance_matrix = np.sum(dr**2, axis=-1)
    return distance_matrix

def test_batching(setup_data):
    """ Test whether batching works as expected by comparing batched and individually generated kNN graphs
    """
    x, = setup_data

    # Test without batch dimension; just pass individual point clouds
    no_batch_results = [jaxkdtree.kNN(x[i], 8, 1000.0) for i in range(x.shape[0])]

    # Test batching 
    batch_results = jax.vmap(jaxkdtree.kNN, in_axes=(0,None,None))(x, 8, 1000.0)

    # Check that results are the same across different batch elements
    for i in range(x.shape[0]):
        assert np.allclose(no_batch_results[i], batch_results[i])

def test_pairwise_dist(setup_data):
    """ Test whether we get the same result using pairwise distances
    """
    x, = setup_data

    # Batched kNN
    batch_results = jax.vmap(jaxkdtree.kNN, in_axes=(0,None,None))(x, 8, 1000.0)

    # Compute pairwise distances and associated NNs
    distance_matrices = jax.vmap(pairwise_distances)(x)
    dist_results = np.argsort(distance_matrices, axis=-1)[..., :8]

    # Check that results are the same across different methods
    assert np.allclose(batch_results, dist_results)
