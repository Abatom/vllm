#!/bin/bash

# Default Configuration
SERVER_IP="0.0.0.0"
SERVER_PORT=10001
INPUT_LENGTH=1024
OUTPUT_LENGTH=1024
BACKEND="vllm"
CARD="H20"
CONCURRENCY_LEVELS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
PROMPTS_PER_CONCURRENCY=10
TOKENIZER_PATH="Llama-3.1-8B-Instruct"
MODEL="base_model"
LOG_FOLDER="auto_benchmark"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --server-ip)
            SERVER_IP="$2"
            shift 2
            ;;
        --server-port)
            SERVER_PORT="$2"
            shift 2
            ;;
        --input-length)
            INPUT_LENGTH="$2"
            shift 2
            ;;
        --output-length)
            OUTPUT_LENGTH="$2"
            shift 2
            ;;
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        --card)
            CARD="$2"
            shift 2
            ;;
        --concurrency-levels)
            # Convert comma-separated values to array
            IFS=',' read -r -a CONCURRENCY_LEVELS <<< "$2"
            shift 2
            ;;
        --prompts-per-concurrency)
            PROMPTS_PER_CONCURRENCY="$2"
            shift 2
            ;;
        --tokenizer-path)
            TOKENIZER_PATH="$2"
            shift 2
            ;;
        --model-name)
            MODEL="$2"
            shift 2
            ;;
        --log-folder)
            LOG_FOLDER="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--server-ip IP] [--server-port PORT] [--input-length LEN] [--output-length LEN]"
            echo "       [--backend BACKEND] [--card CARD] [--concurrency-levels LIST] [--prompts-per-concurrency NUM]"
            echo "       [--tokenizer-path PATH] [--model-name MODEL] [--log-folder FOLDER]"
            exit 1
            ;;
    esac
done

# Current date for log files
CURRENT_DATE=$(date +"%Y_%m_%d_%H_%M")
RESULT_FILE="$LOG_FOLDER/result-${BACKEND}-${CARD}-${INPUT_LENGTH}-${OUTPUT_LENGTH}-${CURRENT_DATE}.txt"

SEED=$(date +%s)
COOLDOWN_PERIOD=1  # cooldown in seconds

# Initialize test environment
mkdir -p "$LOG_FOLDER" || { echo "Failed to create log directory"; exit 1; }

# Create the result file first to avoid tee errors
touch "$RESULT_FILE" || { echo "Failed to create result file"; exit 1; }

# Now we can safely tee to the file
echo "Results will be saved to: $RESULT_FILE" | tee -a "$RESULT_FILE"
echo "Testing configuration:" | tee -a "$RESULT_FILE"
echo "  Model Name: $MODEL_NAME" | tee -a "$RESULT_FILE"
echo "  Server IP: $SERVER_IP" | tee -a "$RESULT_FILE"
echo "  Server Port: $SERVER_PORT" | tee -a "$RESULT_FILE"
echo "  Input Length: $INPUT_LENGTH" | tee -a "$RESULT_FILE"
echo "  Output Length: $OUTPUT_LENGTH" | tee -a "$RESULT_FILE"
echo "  Backend: $BACKEND" | tee -a "$RESULT_FILE"
echo "  Card: $CARD" | tee -a "$RESULT_FILE"
echo "  Concurrency Levels: ${CONCURRENCY_LEVELS[*]}" | tee -a "$RESULT_FILE"
echo "  Prompts per Concurrency: $PROMPTS_PER_CONCURRENCY" | tee -a "$RESULT_FILE"
echo "  Tokenizer Path: $TOKENIZER_PATH" | tee -a "$RESULT_FILE"
echo "  Log Folder: $LOG_FOLDER" | tee -a "$RESULT_FILE"

# Main test execution loop
for concurrency in "${CONCURRENCY_LEVELS[@]}"; do
    total_prompts=$((concurrency * PROMPTS_PER_CONCURRENCY))
    benchmark_log="$LOG_FOLDER/result-${BACKEND}-${CARD}-${INPUT_LENGTH}-${OUTPUT_LENGTH}-${CURRENT_DATE}-${concurrency}.txt"

    echo -e "\n=== Starting test - Concurrency: $concurrency, Total requests: $total_prompts ===" | tee -a "$RESULT_FILE"

    # Execute benchmark test and save to both individual log and result file
    python3 benchmark_serving.py \
        --backend $BACKEND \
        --model $MODEL \
        --tokenizer $TOKENIZER_PATH \
        --dataset-name random \
        --host $SERVER_IP \
        --port $SERVER_PORT \
        --random-input-len $INPUT_LENGTH \
        --random-output-len $OUTPUT_LENGTH \
        --ignore-eos \
        --burstiness 100 \
        --percentile-metrics "ttft,tpot,itl,e2el" \
        --metric-percentiles 90,95,99 \
        --seed $SEED \
        --trust-remote-code \
        --request-rate 1000 \
        --max-concurrency $concurrency \
        --num-prompts $total_prompts \
        --trim-head-ratio 2.0 \
        --trim-tail-ratio 1.0 2>&1 | tee -a "$benchmark_log" "$RESULT_FILE"

    # Verify command execution status
    if [ $? -ne 0 ]; then
        echo "Warning: Test failed at concurrency level $concurrency" | tee -a "$RESULT_FILE"
    fi

    echo "Completed test - Concurrency: $concurrency, Total requests: $total_prompts" | tee -a "$RESULT_FILE"
    echo "Results will be saved to: $RESULT_FILE" | tee -a "$RESULT_FILE"

    # System cooldown period
    echo "Initiating ${COOLDOWN_PERIOD} second cooldown period..." | tee -a "$RESULT_FILE"
    sleep "$COOLDOWN_PERIOD"
done

echo -e "\nAll benchmark tests completed successfully" | tee -a "$RESULT_FILE"

#bash auto_benchmark.sh \
#    --server-ip 0.0.0.0 \
#    --server-port 10001 \
#    --input-length 2048 \
#    --output-length 2048 \
#    --backend vllm \
#    --card H100 \
#    --concurrency-levels 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 \
#    --prompts-per-concurrency 10 \
#    --tokenizer-path /home/Llama-3.1-8B-Instruct \
#    --model-name base_model \
#    --log-folder auto_benchmark