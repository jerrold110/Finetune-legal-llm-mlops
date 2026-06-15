#!/bin/bash

# Set the Tokenizer for TPS metrics
export TOKENIZER="JerroldK/H4-14b-contract-extractor-adapter"

echo "Starting SageMaker endpoint benchmark..."

./awscurl -c 4 -N 10 -X POST \
  -n sagemaker https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/lmi-Hermes-FP8-2026-06-12-15-50-21-251/invocations-response-stream \
  --connect-timeout 120 \
  --dataset prompts \
  -H 'Content-Type: application/json' \
  -H 'X-Amzn-SageMaker-Inference-Component: adapter-lmi-Hermes-FP8-2026-06-12-15-50-21-251' \
  -P -t -o output.txt > benchmark_summary.json

echo "Benchmark complete! Summary is printed above, and raw responses are saved in output.txt.* files."