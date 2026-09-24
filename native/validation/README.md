# Iris native validation

The harness consumes the canonical `Strike1999/Iris` Safetensors snapshot and
uses the released two-stage direct-regression graph: VAE encode, U-Net at
`t=999`, the same U-Net at `t=499`, and VAE decode.

`tools/dump_reference.py` generates deterministic component and full-graph
fixtures from a local snapshot. The ABI and lifecycle smoke tests do not need
model weights. Numerical CPU, Vulkan, Metal, and external-GPU validation must
be recorded here before publishing the runtime in a release catalog.

The production harness keeps the model's processing resolution at its output
boundary. It averages the decoded three-channel disparity image and normalizes
it to a single-channel depth texture without resizing it to capture size.
