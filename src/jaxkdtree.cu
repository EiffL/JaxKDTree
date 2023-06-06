#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "cukd/builder.h"
#include "cukd/knn.h"


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

namespace py = pybind11;

namespace jaxkdtree
{
    /// XLA interface ops
    void kNN(cudaStream_t stream, void **buffers,
            const char *opaque, size_t opaque_len)
    {
        float3 *d_points = (float3 *) buffers[0]; // Input points [N, 3]

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
    m.def("registrations", &jaxkdtree::Registrations);
}