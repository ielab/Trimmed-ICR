#!/bin/bash

# ===== 0.Load environment =====
# For HPC systems, uncomment and modify the following lines:
# module load java/21.0.7
# module load miniconda3
# module load cuda/12.1.1
# module load gcc/12.3.0
# source /path/to/conda.sh
# conda activate /path/to/your/conda/env

# For local systems, activate your conda environment:
# conda activate your_env_name
# Or use virtualenv:
# source your_venv/bin/activate

# Make sure Python, CUDA, and required packages are available in your environment


top_k=100

# ===== 1. Models =====
# LLM_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct
# LLM_NAME=mistralai/Mistral-7B-Instruct-v0.2
LLM_NAME=Qwen/Qwen3-0.6B
# LLM_NAME=Qwen/Qwen3-4B
# LLM_NAME=Qwen/Qwen3-8B


# ===== 2. Run experiments for BEIR Datasets =====
echo "=========================================="
echo "Running experiments for: $LLM_NAME"
echo "=========================================="


# trec-dl datasets
# for data in trec-dl-19 trec-dl-20;

# bright datasets
# for data in aops biology earth_science economics leetcode pony psychology robotics stackoverflow sustainable_living theoremqa_questions theoremqa_theorems;

# beir datasets
for data in climate-fever fiqa fever scidocs trec-covid scifact nfcorpus nq dbpedia-entity;
  do
    echo ""
    echo ">>> Processing dataset: $data with model: $LLM_NAME"
    echo "=========================================="
    
    CUDA_VISIBLE_DEVICES=0 \
    python experiments.py \
        --retriever bm25 \
        --data $data \
        --top_k $top_k \
        --llm_name $LLM_NAME \
        --debug 1000 \
        --seed 0 \
        --reverse_doc_order \
        --reranker icr \
        --truncate_by_space 300 \
        --beir_eval \
        --save_retrieval_results \
        --aggregation_start_layer 10 \
        --max_layer 13 \
        --per_layer_analysis
    
    echo ">>> Completed: $data with $LLM_NAME"
    echo ""
  done

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
