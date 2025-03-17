#include <Python.h>
#include <torch/all.h>
#include <torch/library.h>

#include <vector>
#include "cuda/matrix.cuh"
// #include "cuda/attn.cuh"


extern "C" {
    /* Creates a dummy empty _C module that can be imported from Python.
       The import from Python will load the .so consisting of this file
       in this extension, so that the TORCH_LIBRARY static initializers
       below are run. */
    PyObject* PyInit__C(void)
    {
        static struct PyModuleDef module_def = {
            PyModuleDef_HEAD_INIT,
            "_C",   /* name of module */
            NULL,   /* module documentation, may be NULL */
            -1,     /* size of per-interpreter state of the module,
                       or -1 if the module keeps state in global variables. */
            NULL,   /* methods */
        };
        return PyModule_Create(&module_def);
    }
}

namespace cuda_ext {
TORCH_LIBRARY(cuda_ext, m) {
    m.def("add(Tensor first, Tensor second) -> Tensor");
    m.def("mul(Tensor first, Tensor second) -> Tensor");
    // m.def("attn_score(Tensor Q, Tensor K, Tensor V) -> Tensor");
}

// TORCH_LIBRARY_IMPL(cuda_ext, CPU, m) {
//     m.impl("add", &add);
//     m.impl("mul", &mul);
//     m.impl("attn", &attn);
// }

TORCH_LIBRARY_IMPL(cuda_ext, CUDA, m) {
    m.impl("add", &add);
    m.impl("mul", &mul);
    // m.impl("attn_score", &attn_score);
}

} // namepace cuda_ext