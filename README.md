## Nvidia-tensorrt-llm-API
tensorrt-llm inference api

## TensorRT-LLM Overview

TensorRT-LLM is an easy-to-use Python API to define Large
Language Models (LLMs) and build
[TensorRT](https://developer.nvidia.com/tensorrt) engines that contain
state-of-the-art optimizations to perform inference efficiently on NVIDIA GPUs.
TensorRT-LLM contains components to create Python and C++ runtimes that
execute those TensorRT engines. It also includes a
[backend](https://github.com/triton-inference-server/tensorrtllm_backend)
for integration with the
[NVIDIA Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server);
a production-quality system to serve LLMs.  Models built with TensorRT-LLM can
be executed on a wide range of configurations going from a single GPU to
multiple nodes with multiple GPUs (using
[Tensor Parallelism](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/nlp/nemo_megatron/parallelisms.html#tensor-parallelism)
and/or
[Pipeline Parallelism](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/nlp/nemo_megatron/parallelisms.html#pipeline-parallelism)).

The TensorRT-LLM Python API architecture looks similar to the
[PyTorch](https://pytorch.org) API. It provides a
[functional](./tensorrt_llm/functional.py) module containing functions like
`einsum`, `softmax`, `matmul` or `view`. The [layers](./tensorrt_llm/layers)
module bundles useful building blocks to assemble LLMs; like an `Attention`
block, a `MLP` or the entire `Transformer` layer. Model-specific components,
like `GPTAttention` or `BertAttention`, can be found in the
[models](./tensorrt_llm/models) module.

TensorRT-LLM comes with several popular models pre-defined. They can easily be
modified and extended to fit custom needs. Refer to the [Support Matrix](https://nvidia.github.io/TensorRT-LLM/reference/support-matrix.html) for a list of supported models.

To maximize performance and reduce memory footprint, TensorRT-LLM allows the
models to be executed using different quantization modes (refer to
[`support matrix`](https://nvidia.github.io/TensorRT-LLM/reference/support-matrix.html#software)).  TensorRT-LLM supports
INT4 or INT8 weights (and FP16 activations; a.k.a.  INT4/INT8 weight-only) as
well as a complete implementation of the
[SmoothQuant](https://arxiv.org/abs/2211.10438) technique.

## Getting Started

To get started with TensorRT-LLM, visit our documentation:

- [Quick Start Guide](https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html)
- [Release Notes](https://nvidia.github.io/TensorRT-LLM/release-notes.html)
- [Installation Guide for Linux](https://nvidia.github.io/TensorRT-LLM/installation/linux.html)
- [Installation Guide for Windows](https://nvidia.github.io/TensorRT-LLM/installation/windows.html)
- [Supported Hardware, Models, and other Software](https://nvidia.github.io/TensorRT-LLM/reference/support-matrix.html)

## installation


```bash

# Obtain and start the basic docker image environment (optional).
docker run --rm --runtime=nvidia --gpus all --entrypoint /bin/bash -it nvidia/cuda:12.1.0-devel-ubuntu22.04


```

```bash

# Install dependencies, TensorRT-LLM requires Python 3.10
!apt-get update && apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev git git-lfs

# Install the latest preview version (corresponding to the main branch) of TensorRT-LLM.
# If you want to install the stable version (corresponding to the release branch), please
# remove the `--pre` option.
# use tensorrt_llm = 0.10.0
!pip3 install tensorrt_llm -U  --extra-index-url https://pypi.nvidia.com

# Check installation
!python3 -c "import tensorrt_llm"

```


```bash
# dowmload the model from nvidia NGC catalog
!wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/llama/mistral-7b-int4-chat/versions/1.2/zip -O mistral-7b-int4-chat_1.2.zip

# unzip the model 
unzip mistral-7b-int4-chat_1.2.zip

# clone the nvidia tensorrt llm repo
!git clone -b v0.10.0 https://github.com/NVIDIA/TensorRT-LLM.git

# change directory to the build folder for each model
!cd TensorRT-LLM
!cd examples/llama
```

```bash
# build option 1
!trtllm-build --checkpoint_dir  ./mistral/checkpoint  #directory to the downloaded check point \
              --output_dir      ./mitral/engine_dr # directory to file path of build engine  \
              --gemm_plugin float16 

# build option 2
# build with streaming LLM
!trtllm-build --checkpoint_dir  ./mistral/checkpoint  #directory to the downloaded check point \
              --output_dir      ./mitral/engine_dr # directory to file path of build engine  \
              --gemm_plugin float16 \
              --streamingllm enable

# build option 3
!trtllm-build --checkpoint_dir  ./mistral/checkpoint  #directory to the downloaded check point \
              --output_dir      ./mitral/engine_dr # directory to file path of build engine  \
              --gemm_plugin float16 \
              --streamingllm enable \
              --max_batch_size 8 \
              --max_input_len 1024 \
              --max_output_len 1024 \   
              --tp_size 1 \
              --pp_size 1
```

## Inferencing

```python
from inference_api import TRTLLM_API
```

```python
    ##### replace this with the path of your build trt engine and tokenizer paths of the particular model 
    engine = "path/to/the/mistral build engine"
    token_dr = "path/to/token/dir"

    # create an instance of the model 
    inference_api = TRTLLM_API(engine_dir=engine,token_dir=token_dr)
```

```python
inference_api.generate(input_text= "write a simple python code ",  # input promp
                       streaming=True,                            # streaming output 
                       temperature=1,                             # temperature 
                       max_output_len=500,                        # maximum token output 
                       streaming_interval=1                       # streaming interval if streaming is true
                       )
```

```bash
```