# Third-party notices

The standalone runtime does not dynamically link to or require third-party
libraries other than the operating system's Vulkan loader.

The native runtime is derived from the InferBridge native Lotus harness. Iris
and Lotus are distributed under the Apache License 2.0. The Iris checkpoint
retains the Stable Diffusion 2 VAE and text encoder components described by
the model repository.

`src/shaders/position_bicubic.glsl` and `src/shaders/indexing.h` are adapted
from the PyTorch Vulkan backend, Copyright (c) Facebook, Inc. and its
affiliates, under PyTorch's BSD-style license:

https://github.com/pytorch/pytorch/blob/main/LICENSE
