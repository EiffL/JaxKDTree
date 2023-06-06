#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "cukd/builder.h"
#include "cukd/knn.h"

template <typename T>
pybind11::capsule EncapsulateFunction(T* fn) {
  return pybind11::capsule(bit_cast<void*>(fn), "xla._CUSTOM_CALL_TARGET");
}


namespace py = pybind11;
namespace jkd = jaxkdtree;

namespace jaxkdtree
{
    /// XLA interface ops
    void kNN(cudaStream_t stream, void **buffers,
            const char *opaque, size_t opaque_len)
    {
        void *points_d = buffers[0]; // Input points [N, 3]

        // TODO decompress opaque to know the number of points
        int nPoints = 100;

        // Build the KDTree from the provided points
        cukd::buildTree<cukd::TrivialFloatPointTraits<float3>>(d_points, nPoints, stream);
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
    m.def("registrations", &jkd::Registrations);
}