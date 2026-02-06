#!/bin/bash

module load java/21.0.7 miniconda3 cuda/12.1.1 gcc/12.3.0
source /sw/auto/rocky8d/epyc3_a100/software/Miniconda3/23.9.0-0/etc/profile.d/conda.sh
conda activate /scratch/user/uqdche12/envs/setwise-icr
export PATH="/scratch/user/uqdche12/envs/setwise-icr/bin:$PATH"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
cd /scratch/user/uqdche12/code/llm-rankers
set -euo pipefail

DATASET=${1:-trec-covid}
MODEL_NAME=${2:-meta-llama/Llama-3.1-8B-Instruct}
MODE=${3:-attention}
WINDOW_SIZE=${4:-100} # default: 100
STEP_SIZE=${5:-1} # default: 1
NUM_REPEAT=${6:-1} # default: 1
MAX_LAYER=${7:-}  # default: None
AGGREGATION_START_LAYER=${8:-}  # default: None
SCORING_STRATEGY=${9:-masked_NA_calibration}  #default: masked_NA_calibration

echo "Running BEIR Listwise experiment on dataset: ${DATASET}"
echo "Using model: ${MODEL_NAME}"
echo "Mode: ${MODE}"
echo "Window size: ${WINDOW_SIZE}"
echo "Step size: ${STEP_SIZE}"
echo "Num repeat: ${NUM_REPEAT}"
if [ -n "${MAX_LAYER}" ]; then
  echo "Max layer: ${MAX_LAYER}"
fi
if [ -n "${AGGREGATION_START_LAYER}" ]; then
  echo "Aggregation start layer: ${AGGREGATION_START_LAYER}"
fi
if [ "${MODE}" = "attention" ]; then
  echo "Scoring strategy: ${SCORING_STRATEGY}"
fi
echo "=========================================="

# Step 1: BM25 first-stage
BM25_RUN="run.bm25.${DATASET}.txt"
if [ ! -f "$BM25_RUN" ]; then
    echo "Generating BM25 run..."
    python -m pyserini.search.lucene \
      --index beir-v1.0.0-${DATASET}.flat \
      --topics beir-v1.0.0-${DATASET}-test \
      --output ${BM25_RUN} \
      --output-format trec \
      --batch 36 --threads 12 \
      --hits 100 --bm25 --remove-query
fi

# Step 2: Listwise reranking
MODEL_TAG="${MODEL_NAME//\//__}"
MODEL_TAG="${MODEL_TAG//:/__}"

SCORING_ARGS=""
LAYER_SUFFIX=""
if [ "${MODE}" = "attention" ]; then
  RETRIEVAL_TYPE="IE"
  if [ "${DATASET}" = "trec-covid" ] || [ "${DATASET}" = "fiqa" ] || [ "${DATASET}" = "webis-touche2020" ] || [ "${DATASET}" = "dbpedia-entity" ] || [ "${DATASET}" = "nq" ]; then
    RETRIEVAL_TYPE="QA"
  fi
  SCORING_ARGS="--scoring attention --use_fa2 --retrieval_type ${RETRIEVAL_TYPE} --prompt_template instruct --scoring_strategy ${SCORING_STRATEGY} --reverse_doc_order"
  case "${SCORING_STRATEGY}" in
    masked_NA_calibration) SCORING_ABBREV="mNAcal" ;;
    query_last)            SCORING_ABBREV="qlast" ;;
    attention_sorting)      SCORING_ABBREV="attn_sort" ;;
    NA_only)               SCORING_ABBREV="NAonly" ;;
    NA_calibration_no_agg) SCORING_ABBREV="NAcal_noagg" ;;
    *)                     SCORING_ABBREV="${SCORING_STRATEGY}" ;;
  esac
  LAYER_SUFFIX="${LAYER_SUFFIX}.${SCORING_ABBREV}"
  if [ -n "${MAX_LAYER}" ]; then
    SCORING_ARGS="${SCORING_ARGS} --max_layer ${MAX_LAYER}"
    LAYER_SUFFIX="${LAYER_SUFFIX}.maxL${MAX_LAYER}"
  fi
  if [ -n "${AGGREGATION_START_LAYER}" ]; then
    SCORING_ARGS="${SCORING_ARGS} --aggregation_start_layer ${AGGREGATION_START_LAYER}"
    LAYER_SUFFIX="${LAYER_SUFFIX}.aggL${AGGREGATION_START_LAYER}"
  fi
elif [ "${MODE}" = "likelihood" ]; then
  SCORING_ARGS="--scoring likelihood"
else
  SCORING_ARGS="--scoring generation"
fi

LISTWISE_RUN="run.listwise.${DATASET}.${MODEL_TAG}.${MODE}.w${WINDOW_SIZE}.s${STEP_SIZE}.r${NUM_REPEAT}${LAYER_SUFFIX}.txt"
echo "Running listwise reranking with scoring=${MODE}, window_size=${WINDOW_SIZE}, step_size=${STEP_SIZE}, num_repeat=${NUM_REPEAT}..."

python run.py \
  run --model_name_or_path ${MODEL_NAME} \
      --tokenizer_name_or_path ${MODEL_NAME} \
      --run_path ${BM25_RUN} \
      --save_path ${LISTWISE_RUN} \
      --pyserini_index beir-v1.0.0-${DATASET} \
      --hits 100 \
      --query_length 32 \
      --passage_length 128 \
      ${SCORING_ARGS} \
      --device cuda \
  listwise --window_size ${WINDOW_SIZE} \
           --step_size ${STEP_SIZE} \
           --num_repeat ${NUM_REPEAT}

# Step 3: Evaluation
echo "Evaluating results..."

# Step 4: Save results
mkdir -p output

EVAL_OUTPUT=$(python -m pyserini.eval.trec_eval \
  -c -m ndcg_cut.10 beir-v1.0.0-${DATASET}-test \
  ${LISTWISE_RUN} 2>&1)

echo "$EVAL_OUTPUT"

EVAL_RESULTS=$(echo "$EVAL_OUTPUT" | grep -A 1 "^Results:" | grep -E "(^Results:|^ndcg_cut_10)")

NDCG_SCORE=$(echo "$EVAL_OUTPUT" | grep "ndcg_cut_10" | awk '{print $3}')

MODEL_TAG="${MODEL_NAME//\//__}"
MODEL_TAG="${MODEL_TAG//:/__}"

abbrev_mode() {
  case "$1" in
    attention) echo "attn" ;;
    generation) echo "gen" ;;
    likelihood) echo "lik" ;;
    *) echo "$1" ;;
  esac
}

MODE_ABBREV=$(abbrev_mode "${MODE}")

STATS_FILENAME_PARTS=("${DATASET}" "${MODEL_TAG}" "${MODE_ABBREV}" "w${WINDOW_SIZE}" "s${STEP_SIZE}" "r${NUM_REPEAT}")
if [ "${MODE}" = "attention" ]; then
  RETRIEVAL_TYPE="IE"
  if [ "${DATASET}" = "trec-covid" ] || [ "${DATASET}" = "fiqa" ] || [ "${DATASET}" = "webis-touche2020" ] || [ "${DATASET}" = "dbpedia-entity" ] || [ "${DATASET}" = "nq" ]; then
    RETRIEVAL_TYPE="QA"
  fi
  STATS_FILENAME_PARTS+=("${SCORING_ABBREV}" "${RETRIEVAL_TYPE}" "instr" "rev")
  if [ -n "${MAX_LAYER}" ]; then
    STATS_FILENAME_PARTS+=("maxL${MAX_LAYER}")
  fi
  if [ -n "${AGGREGATION_START_LAYER}" ]; then
    STATS_FILENAME_PARTS+=("aggL${AGGREGATION_START_LAYER}")
  fi
fi
STATS_FILENAME=$(IFS='_'; echo "${STATS_FILENAME_PARTS[*]}")_stats.txt
STATS_FILE="output/${STATS_FILENAME}"

sleep 1

echo "Looking for statistics file: ${STATS_FILE}"
if [ -f "$STATS_FILE" ]; then
  echo "Found statistics file, appending evaluation results..."
  echo "" >> "$STATS_FILE"
  echo "$EVAL_RESULTS" >> "$STATS_FILE"
  echo "Complete statistics saved to: ${STATS_FILE}"
else
  echo "Warning: Statistics file not found: ${STATS_FILE}"
  echo "Available files in output directory:"
  ls -la output/ | grep "${DATASET}" | grep "${MODEL_TAG}" || echo "No matching files found"
fi

echo "Done! Output: ${LISTWISE_RUN}"
