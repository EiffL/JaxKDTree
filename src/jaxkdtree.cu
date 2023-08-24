#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "cukd/builder.h"
#include "cukd/knn.h"


namespace py = pybind11;

// https://en.cppreference.com/w/cpp/numeric/bit_cast
template <class To, class From>
typename std::enable_if<sizeof(To) == sizeof(From) && std::is_trivially_copyable<From>::value &&
                            std::is_trivially_copyable<To>::value,
                        To>::type
bit_cast(const From& src) noexcept {
static_assert(
    std::is_trivially_constructible<To>::value,
    "This implementation additionally requires destination type to be trivially constructible");

To dst;
memcpy(&dst, &src, sizeof(To));
return dst;
}


template <typename T>
std::string PackDescriptorAsString(const T& descriptor) {
  return std::string(bit_cast<const char*>(&descriptor), sizeof(T));
}

template <typename T>
pybind11::bytes PackDescriptor(const T& descriptor) {
  return pybind11::bytes(PackDescriptorAsString(descriptor));
}

template <typename T>
    const T* UnpackDescriptor(const char* opaque, std::size_t opaque_len) {
    if (opaque_len != sizeof(T)) {
        throw std::runtime_error("Invalid opaque object size");
    }
    return bit_cast<const T*>(opaque);
    }

template <typename T>
pybind11::capsule EncapsulateFunction(T* fn) {
  return pybind11::capsule(bit_cast<void*>(fn), "xla._CUSTOM_CALL_TARGET");
}


// ==================================================================
__global__ void d_knn(int32_t *d_results,
                       float3 *d_queries,
                       int numBatches,
                       int numQueries,
                       float3 *d_nodes,
                       int numNodes,
                       float maxRadius)
{
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  int bid = blockIdx.y;
  if (tid >= numQueries) return;
  const int k = 8;

  cukd::FixedCandidateList<k> result(maxRadius);
  float sqrDist
    = cukd::knn
    <cukd::TrivialFloatPointTraits<float3>>
    (result, d_queries[bid * numQueries + tid], d_nodes, numNodes);

  for(int i = 0; i < k; i++){
      d_results[bid * numQueries * k + tid * k + i] = result.decode_pointID(result.entry[i]);
  };

}

void knn(int32_t *d_results,
         float3 *d_queries,
         int numBatches,
         int numQueries,
         float3 *d_nodes,
         int numNodes,
         float maxRadius,
         cudaStream_t stream)
{
  dim3 bs(128, 1, 1);  // Block size remains the same in the x dimension
  dim3 grid(cukd::common::divRoundUp(static_cast<uint32_t>(numQueries), static_cast<uint32_t>(bs.x)), numBatches);

  d_knn<<<grid, bs, 0, stream>>>(d_results, d_queries, numBatches, numQueries, d_nodes, numNodes, maxRadius);
}


namespace jaxkdtree
{

    struct kNNDescriptor {
      int nPoints;
      int k;
      double radius;
      int numBatches;
    };

    /// XLA interface ops
    void kNN(cudaStream_t stream, void **buffers,
         const char *opaque, size_t opaque_len)
    {
        float3 *d_points = (float3 *) buffers[0];
        float3 *d_queries = d_points;
        int32_t* d_results = (int32_t *) buffers[1];

        const kNNDescriptor &d =
          *UnpackDescriptor<kNNDescriptor>(opaque, opaque_len);

        int numBatches = d.numBatches;

        // Loop over batches, construct KDTree and perform kNN search
        for (int batchIdx = 0; batchIdx < numBatches; ++batchIdx) {

            // Construct the KDTree for the current batch
            cukd::buildTree<cukd::TrivialFloatPointTraits<float3>>(d_points + batchIdx * d.nPoints, d.nPoints, stream);

            // Perform the kNN search for the current batch
            knn(d_results + batchIdx * d.nPoints * d.k, d_queries + batchIdx * d.nPoints, 1, d.nPoints, d_points + batchIdx * d.nPoints, d.nPoints, d.radius, stream);
        }

        // // Build the KDTree from the provided points
        // cukd::buildTree<cukd::TrivialFloatPointTraits<float3>>(d_points, d.nPoints, stream);

        // // Perform the kNN search
        // knn(d_results, d_queries, numBatches, d.nPoints, d_points, d.nPoints, d.radius, stream);
    }

    // Utility to export ops to XLA
    py::dict Registrations()
    {
        py::dict dict;
        dict["kNN"] = EncapsulateFunction(kNN);
        return dict;
    }
}

PYBIND11_MODULE(_jaxkdtree, m)
{
    // Function registering the custom ops
    m.def("registrations", &jaxkdtree::Registrations);
    m.def("create_kNN_descriptor", [](int nPoints, int k, double radius, int numBatches) {
      return PackDescriptor(jaxkdtree::kNNDescriptor{
                nPoints, k, radius, numBatches});
      });
}