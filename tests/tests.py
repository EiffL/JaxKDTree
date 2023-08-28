import jax
import jax.numpy as np
import jaxkdtree
import pytest

# Constants
N_NODES = 1000
N_BATCH = 32
K_NEAREST_NEIGHBORS = 8
DISTANCE_THRESHOLD = 1000.0

@pytest.fixture
def setup_data():
    """ Generate a random point cloud """
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (N_BATCH, N_NODES, 3))
    return x,

def pairwise_distances(point_cloud):
    """ Compute pairwise distances between points in a point cloud. """
    dr = point_cloud[:, None, :] - point_cloud[None, :, :]
    return np.sum(dr**2, axis=-1)

def extract_distances_from_indices(distance_matrices, indices):
    """ Extract distances using indices from the distance matrices. """
    n_batch, n_points, _ = distance_matrices.shape
    rows = np.arange(n_batch)[:, None, None]
    columns = np.arange(n_points)[:, None]
    return distance_matrices[rows, columns, indices]

def test_batching(setup_data):
    """ Test batching vs individual kNN graphs """
    x, = setup_data

    # Individual kNN results
    no_batch_results = [jaxkdtree.kNN(x[i], K_NEAREST_NEIGHBORS, DISTANCE_THRESHOLD) for i in range(N_BATCH)]

    # Batched kNN results
    batch_results = jax.vmap(jaxkdtree.kNN, in_axes=(0,None,None))(x, K_NEAREST_NEIGHBORS, DISTANCE_THRESHOLD)

    # Comparing individual and batched results
    for i in range(N_BATCH):
        assert np.allclose(no_batch_results[i], batch_results[i])

def test_pairwise_dist(setup_data):
    """ Test kNN vs pairwise distances method """
    x, = setup_data

    # Batched kNN results
    batch_results = jax.vmap(jaxkdtree.kNN, in_axes=(0,None,None))(x, K_NEAREST_NEIGHBORS, DISTANCE_THRESHOLD)

    # Pairwise distances and sorted indices
    distance_matrices = jax.vmap(pairwise_distances)(x)
    dist_results_indices = np.argsort(distance_matrices, axis=-1)[..., :K_NEAREST_NEIGHBORS]

    # Extracting distances using indices
    dist_distances = extract_distances_from_indices(distance_matrices, dist_results_indices)
    batch_distances = extract_distances_from_indices(distance_matrices, batch_results)

    # Comparing results
    assert np.allclose(batch_results, dist_results_indices)
    assert np.allclose(batch_distances, dist_distances)
