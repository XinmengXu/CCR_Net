# Implementation scope

The repository implements the method described in the paper through native
PyTorch modules and a repository structure designed for this project. It does
not include source files from an external AVSS framework.

The implementation uses common mathematical and software building blocks such
as STFT/iSTFT, convolution, attention, singular-value decomposition, AdamW, and
JSONL manifests. These operations are provided by third-party libraries under
their own licenses.

Any future addition copied or adapted from another repository must retain the
license, copyright notice, and attribution required by that source.
