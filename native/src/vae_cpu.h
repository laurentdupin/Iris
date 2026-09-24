#pragma once

#include "tensor_cpu.h"

namespace iris_native {

struct Posterior {
    ImageTensor mean;
    ImageTensor log_variance;
};

Posterior vae_encode(
    const SafeTensors& model,
    const ImageTensor& rgb);
ImageTensor vae_decode(
    const SafeTensors& model,
    const ImageTensor& latent);

}  // namespace iris_native
