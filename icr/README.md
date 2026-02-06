# Reproduce: In-Context-Reranking
 **Original repo**: [OSU-NLP-Group/In-Context-Reranking](https://github.com/OSU-NLP-Group/In-Context-Reranking)

 **Original paper**: [Attention in Large Language Models Yields Efficient Zero-Shot Re-Rankers](https://arxiv.org/abs/2410.02642)

## Installation

**Option 1: Using conda (recommended)**
```bash
conda env create -f environment.yml
conda activate icr
```

**Option 2: Using pip**
```bash
pip install -r requirements.txt
```

**Flash-Attention**
If you need flash attention support (for faster inference), install a pre-built wheel from [Flash-Attention releases](https://github.com/Dao-AILab/flash-attention/releases). Choose the wheel that matches your CUDA version, PyTorch version, and Python version. For example:
```bash
# For CUDA 12.3, PyTorch 2.4, Python 3.10:
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abifalse-cp310-cp310-linux_x86_64.whl
```

**Note:** For BM25 retrieval, you also need to install [Pyserini](https://github.com/castorini/pyserini) and Java.

## Data Preparation

You can either generate retrieval results yourself or download pre-computed results:

**Option 1: Generate retrieval results**
- For BEIR and TREC datasets: Run `src/bm25_retrieval.py` (requires [Pyserini](https://github.com/castorini/pyserini))
- For BRIGHT datasets: Run `src/bright_bm25_retrieval.py`
- Results will be stored in `retriever_outpout/` with format `icr_{type}_{dataset}_{retriever}_top_{k}.json`

**Option 2: Download pre-computed results**
- Download retrieval results from [Google Drive](https://drive.google.com/file/d/1jleC9MeUkSl2MN6OG1rTzze-dFA8KMDV/view) and put them in `retriever_outpout/`

### Custom dataset
Process your own data into the following json format:
```json
[
  {
	"idx": "idx will be used to retrieve qrel records",
	"question": "query for retrieval or QA",
	"paragraphs":[
	  {
	    "idx": "idx of documents",
		"title": "title of document",
		"paragraph_text": "text of document",
		"is supporting": "true/false, whether the document is a target for retrieval",
	  },
	  {},
	]
  },
  {},
]
```
## Experiments
We provide the scripts for reproducing our experiments:

```bash
bash run_batch_icr.sh
```

### Key Configuration Options

**Required arguments:**
- `--llm_name`: HuggingFace model name (e.g., `Qwen/Qwen3-0.6B`)
- `--data`: Dataset name (e.g., `aops`, `trec-covid`, `theoremqa_theorems`)
- `--reranker`: Reranker type (`icr`)

**Retrieval settings:**
- `--retriever`: Base retriever (`bm25`)
- `--top_k`: Number of documents to retrieve (default: 20)

**ICR-specific options:**
- `--scoring_strategy`: Scoring method (default: `masked_NA_calibration`)
- `--reverse_doc_order`: Reverse document order (most relevant at the end)
- `--truncate_by_space`: Truncate documents to N words (e.g., 300)
- `--aggregation_start_layer`: Start layer for attention aggregation (0-indexed)
- `--max_layer`: Maximum layer to compute (0-indexed)

**Evaluation options:**
- `--save_retrieval_results`: Save retrieval results to JSON
- `--per_layer_analysis`: Compute metrics for each layer separately

## Citation
If you find this work helpful, please consider citing our paper:
```
@misc{chen2024attentionlargelanguagemodels,
      title={Attention in Large Language Models Yields Efficient Zero-Shot Re-Rankers}, 
      author={Shijie Chen and Bernal Jiménez Gutiérrez and Yu Su},
      year={2024},
      eprint={2410.02642},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.02642}, 
}
```