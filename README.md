# AutoSP Setup Guide

Quick start guide to clone and set up the AutoSP repository.

## Prerequisites

- CUDA 12.8 compatible GPU (recommended)
- Conda installed
- Git

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/spikerheado1234/AutoSP.git
cd AutoSP
```

### 2. Create Conda Environment

```bash
conda create --prefix ./autosp_env python=3.10 -y
conda activate ./autosp_env
```

### 3. Install PyTorch

```bash
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 4. Install Dependencies

```bash
pip install \
  transformers==4.50.3 \
  tokenizers==0.15.2 \
  huggingface-hub==0.25.1 \
  safetensors==0.4.5 \
  datasets \
  accelerate \
  scipy \
  tqdm \
  pyyaml
```

### 5. Install AutoSP

```bash
pip install --no-build-isolation -e .
```


## Benchmarking

See `bench_dc_ulysses/` directory for benchmarking scripts:

```bash
cd bench_dc_ulysses
bash run_ulysses.sh {seq_len} {compile|eager|deepcompile} {num_layers}
```

## Troubleshooting

- **Build isolation errors**: Use `pip install --no-build-isolation -e .`
- **CUDA issues**: Verify CUDA 12.8 compatibility with `nvidia-smi`
- **Out of memory**: Adjust batch size or sequence length parameters
