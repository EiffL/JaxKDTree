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
__global__ void d_knn(uint32_t *d_results,
                       float3 *d_queries,
                       int numQueries,
                       float3 *d_nodes,
                       int numNodes,
                       float maxRadius)
{
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  if (tid >= numQueries) return;

  cukd::FixedCandidateList<3> result(maxRadius);
  float sqrDist
    = cukd::knn
    <cukd::TrivialFloatPointTraits<float3>>
    (result,d_queries[tid],d_nodes,numNodes);

  for(int i=0; i < 3; i++){
    d_results[tid*3+i] = result.decode_pointID(result.entry[i]);
 };

}

void knn(uint32_t *d_results,
          float3 *d_queries,
          int numQueries,
          float3 *d_nodes,
          int numNodes,
          float maxRadius)
{
  int bs = 128;
  int nb = cukd::common::divRoundUp(numQueries,bs);
  d_knn<<<nb,bs>>>(d_results,d_queries,numQueries,d_nodes,numNodes,maxRadius);
}


namespace jaxkdtree
{
    /// XLA interface ops
    void kNN(cudaStream_t stream, void **buffers,
            const char *opaque, size_t opaque_len)
    {
        float3 *d_points = (float3 *) buffers[0]; // Input points [N, 3]
        float3 *d_queries = (float3 *) buffers[1]; // Input query [N, 3]
        uint32_t* d_results = (uint32_t *) buffers[2]; // Output buffer [N, k]

        // TODO decompress opaque to know the number of points
        int nPoints = 100;

        // Build the KDTree from the provided points
        cukd::buildTree<cukd::TrivialFloatPointTraits<float3>>(d_points, nPoints, stream);

        // Perform the kNN search
        knn(d_results, d_queries, nPoints, d_points, nPoints, 10.);
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
}