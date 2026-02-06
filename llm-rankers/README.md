# Reproduce: llm-rankers
 **Original repo**: [ielab/llm-rankers](https://github.com/ielab/llm-rankers)

 **Original paper**: [A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking with Large Language Models](https://arxiv.org/pdf/2310.09497.pdf)


## Prerequisites
This package requires an existing ICR environment (see [ICR repository](https://github.com/your-icr-repo) for setup instructions).

This codebase uses ICR (In-Context Reranking) as a scoring strategy (e.g., `--scoring attention`), which requires ICR's modules and dependencies. Therefore, you need to:
1. First install ICR and its dependencies (torch, transformers, flash-attn, etc.)
2. Then install llm-rankers dependencies on top of the ICR environment

## Installation

1. Activate your ICR conda environment:
```bash
conda activate your_icr_env
```

2. Clone this repo and install the package (this will automatically install all dependencies):
```bash
git clone https://github.com/ielab/llm-rankers.git
cd llm-rankers
pip install -e .
```

**Note:** The code is tested with python=3.10 conda environment. You may also need to install some pyserini dependencies such as Java. We refer to pyserini installation doc [link](https://github.com/castorini/pyserini/blob/master/docs/installation.md#development-installation)



## Experiments (BEIR)

We provide convenient scripts to run experiments on BEIR datasets. The scripts automatically handle BM25 first-stage retrieval, LLM re-ranking, and evaluation.

### Listwise Re-ranking

Run listwise re-ranking experiments:
```bash
bash run_beir_listwise.sh [dataset] [model_name] [mode] [window_size] [step_size] [num_repeat] [max_layer] [aggregation_start_layer] [scoring_strategy]
```

**Parameters:**
- `dataset`: BEIR dataset name (default: `trec-covid`)
- `model_name`: HuggingFace model name (default: `meta-llama/Llama-3.1-8B-Instruct`)
- `mode`: Scoring mode - `attention`, `generation`, or `likelihood` (default: `attention`)
- `window_size`: Sliding window size (default: `100`)
- `step_size`: Step size for sliding window (default: `1`)
- `num_repeat`: Number of repeats (default: `1`)
- `max_layer`: Maximum layer for attention aggregation (optional)
- `aggregation_start_layer`: Start layer for aggregation (optional)
- `scoring_strategy`: Scoring strategy for attention mode (default: `masked_NA_calibration`)

**Example:**
```bash
bash run_beir_listwise.sh trec-covid meta-llama/Llama-3.1-8B-Instruct attention 100 1 1
```

### Setwise Re-ranking

Run setwise re-ranking experiments:
```bash
bash run_beir_setwise.sh [dataset] [model_name] [mode] [method] [max_layer] [aggregation_start_layer] [scoring_strategy]
```

**Parameters:**
- `dataset`: BEIR dataset name (default: `nfcorpus`)
- `model_name`: HuggingFace model name (default: `meta-llama/Llama-3.1-8B-Instruct`)
- `mode`: Scoring mode - `attention`, `generation`, or `likelihood` (default: `generation`)
- `method`: Sorting method - `heapsort` or `bubblesort` (default: `heapsort`)
- `max_layer`: Maximum layer for attention aggregation (optional)
- `aggregation_start_layer`: Start layer for aggregation (optional)
- `scoring_strategy`: Scoring strategy for attention mode (default: `masked_NA_calibration`)

**Example:**
```bash
bash run_beir_setwise.sh nfcorpus meta-llama/Llama-3.1-8B-Instruct generation heapsort
```

The scripts automatically:
1. Generate BM25 first-stage results (if not already present)
2. Run LLM re-ranking with specified parameters
3. Evaluate results using NDCG@10
4. Save statistics to `output/` directory