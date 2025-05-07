# 🧠 GPT-2 Attention Block in JAX + Docker GPU Exploration

### A mini-project by **Shushawn Saha**

---

## 📌 Overview

This project explores the inner workings of the GPT-2 attention mechanism, focusing on converting its PyTorch implementation from [nanoGPT](https://github.com/karpathy/nanoGPT) to [JAX](https://github.com/google/jax) using the **Flax** library. In addition, it dives into how **Flash Attention** can optimize memory and compute efficiency. The second part explores the NVIDIA CUDA Docker image, including environment setup and GPU accessibility without Docker or a physical GPU.

---

## Part 1: 🚀 GPT-2 Attention Block in JAX

### 🔍 Objective

Reimplement the GPT-2 attention block using **JAX** while maintaining functional parity with the PyTorch version from nanoGPT. Extend this implementation by investigating **Flash Attention** for performance optimization.

### 🔧 What is Attention?

Attention allows a model to weigh the importance of different input tokens. Each token gets three vectors:
- **Query (q)**: What the word is looking for.
- **Key (k)**: What the word has to offer.
- **Value (v)**: The actual content.

The process:
1. Compute dot products of queries with keys.
2. Apply softmax to get attention scores.
3. Use scores to perform a weighted sum of the values.

### 🧱 From PyTorch to JAX

- **Original Codebase**: [nanoGPT model.py](https://github.com/karpathy/nanoGPT/blob/master/model.py#L29)
- Key operations translated:
  - Multi-head attention using matrix manipulation
  - Causal masking to prevent access to future tokens
  - Attention and residual dropout layers

### 🧪 Testing

- ✅ Matched output **shapes** for PyTorch and JAX
- ❌ Max absolute difference in outputs > 1e-5  
  - Potential reason: different softmax handling in JAX vs PyTorch

### ⚡ Flash Attention

- **Goal**: Reduce memory usage & improve speed without compromising accuracy
- **Techniques**:
  - **Tiling**: Compute attention in chunks to better use memory
  - **Fused Kernels**: Combine operations for efficiency
  - **Recomputation**: Save compute over memory

#### JAX Application:
Use the built-in Flash Attention with:
```python
y = flash_attention(q, k, v, causal=True)
```

---

## Part 2: 🐳 Docker, CUDA & GPUs (Without GPUs)

### 🎯 Docker Image Used

- **Base Image**: `nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04`
- **Manifest SHA**: `sha256:ea73ae92d1ab9453de0910d342b005aaec8fa2388d3f8913694a6de69392c6ab`
- **Top Layer SHA**: `sha256:687d50f2f6a697da02e05f2b2b9cb05c1d551f37c404ebe55fdec44b0ae8aa5c`

### 🧪 Exploring the Layer

- Tried using both `curl` and `docker save` to download and unpack the layer
- Encountered system crashes due to the large 1.1GB layer

### 📁 Layer Details

- Inspected via `dive`
- Contains typical CUDA & cuDNN development files under `/usr`

### 🏃‍♂️ Running the Container

```bash
docker run --rm -it nvcr.io/nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04
```

### 🔍 Environment Variables

- `LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64`

### 📦 GPU Libraries in Path

- ✅ `libcuda`, `libcufft` found via `ldconfig`

### 🔌 Role of NVIDIA Container Toolkit

- Enables Docker containers to access host GPUs
- Required for `nvidia-docker` and proper GPU driver integration

---

## 🛠️ Setup & Run

### 🔧 Requirements

- Python (3.8+)
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### ▶️ Run

```bash
python main.py
```

---

## 🧠 Reflections

- Matching outputs exactly in JAX and PyTorch can be tricky due to backend-level differences in operations like softmax.
- Flash Attention shows promise for speed and memory, but implementation details (like tiling) matter.
- Working with Docker layers gives insight into the inner workings of CUDA containers—though hardware limitations can pose real obstacles.

---

## 📂 Project Structure

```
.
├── attention_jax.py       # JAX implementation of GPT-2 attention
├── main.py                # Driver code for testing
├── requirements.txt       # Python dependencies
└── README.md              # This file
```
