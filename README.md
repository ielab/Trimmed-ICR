# Where Relevance Emerges - A Layer-Wise Study of Internal Attention for Zero-Shot Re-Ranking

This repository contains reproduction code for paper: **"Where Relevance Emerges: A Layer-Wise Study of Internal Attention for Zero-Shot Re-Ranking"**.

## Overview

This work investigates how relevance signals are distributed across transformer layers in Large Language Models (LLMs) for zero-shot document re-ranking. The main contributions include:

1. **Layer-wise Analysis**: Discovering a universal "Bell-Curve" distribution of relevance signals across transformer layers
2. **Trimmed-ICR**: A strategy that reduces inference latency by 30%-50% without compromising effectiveness by focusing on high-signal layers
3. **Unified Comparison**: Systematic evaluation of three scoring mechanisms (generation, likelihood, internal attention) across Listwise and Setwise ranking frameworks
4. **BRIGHT Benchmark**: Demonstrating that attention-based scoring enables small models (0.6B) to outperform GPT-4-based generative re-rankers

## Repository Structure

This codebase reproduces two main components:

### 1. `icr/` - In-Context Reranking (ICR)
- **Original repo**: [OSU-NLP-Group/In-Context-Reranking](https://github.com/OSU-NLP-Group/In-Context-Reranking)
- **Original paper**: [Attention in Large Language Models Yields Efficient Zero-Shot Re-Rankers](https://arxiv.org/abs/2410.02642)
- **Features**:
  - Implements ICR method using internal attention signals
  - Supports layer-wise analysis and per-layer evaluation
  - Implements Trimmed-ICR strategy (layer selection and aggregation)
  - Evaluates on TREC DL, BEIR, and BRIGHT datasets

### 2. `llm-rankers/` - LLM-based Ranking Methods
- **Original repo**: [ielab/llm-rankers](https://github.com/ielab/llm-rankers)
- **Original paper**: [A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking with Large Language Models](https://arxiv.org/pdf/2310.09497.pdf)
- **Features**:
  - Implements Pointwise, Listwise, Pairwise, and Setwise ranking methods
  - Supports three scoring mechanisms: `generation`, `likelihood`, and `attention` (using ICR)
  - Enables unified comparison of different scoring strategies under consistent conditions
  - Evaluates on BEIR datasets

## Key Research Questions

- **RQ1**: How is the relevance signal distributed across transformer layers, and do all layers contribute equally?
- **RQ2**: How does attention-based scoring compare with generative and likelihood-based methods in Listwise and Setwise frameworks?
- **RQ3**: Does attention-based ranking remain effective on reasoning-intensive tasks, and is the layer-wise signal distribution universal?

## Quick Start

See individual README files for detailed installation and usage:
- [`icr/README.md`](icr/README.md) - ICR implementation and experiments
- [`llm-rankers/README.md`](llm-rankers/README.md) - Ranking methods and scoring comparison

## Citation

If you use this code, please cite the original papers:

```bibtex
TBD
```
