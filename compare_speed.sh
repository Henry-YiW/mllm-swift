#!/bin/bash

# Define variables for better readability and maintenance
SWIFT_OUTPUT="/scratch/bdes/haorany7/swift/Henry/mllm-swift/outputs/llava/llava_100/model_answer/llava-1.5-13b-hf/llava-1.5-13b-hf-swift-float16-temp-0.2-top-p-0.85-seed-2024-max_new_tokens-512-opt_interval-1-bayes_interval-25-max_opt-1000-max_tolerance-300-max_score-0.93-context_window-50-skip_ratio-0.0.jsonl"
BASE_OUTPUT="/scratch/bdes/haorany7/swift/Henry/mllm-swift/test/llava/llava_100/model_answer/llava-1.5-13b-hf/llava-1.5-13b-hf-vanilla-float16-temp-0.2-top-p-0.85-seed-2024-max_new_tokens-512.jsonl"
TOKENIZER="llava-hf/llava-1.5-13b-hf"
# Run the evaluation
python evaluation_llama/speed.py \
    --file-path "$SWIFT_OUTPUT" \
    --base-path "$BASE_OUTPUT" \
    --tokenizer-path "$TOKENIZER"