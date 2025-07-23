#!/bin/bash

# Default Configuration
SERVER_IP="0.0.0.0"
SERVER_PORT=10001
INPUT_LENGTH=1024
OUTPUT_LENGTH=1024
BACKEND="vllm"
CARD="H20"
QPS_LEVELS=(1 2 3 4 5 6 7 8 9 10)
PROMPTS_PER_QPS=300
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
        --qps-levels)
            # Convert comma-separated values to array
            IFS=',' read -r -a QPS_LEVELS <<< "$2"
            shift 2
            ;;
        --prompts-per-qps)
            PROMPTS_PER_QPS="$2"
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
            echo "       [--backend BACKEND] [--card CARD] [--qps-levels LIST] [--prompts-per-qps NUM]"
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
echo "  QPS Levels: ${QPS_LEVELS[*]}" | tee -a "$RESULT_FILE"
echo "  Prompts per QPS: $PROMPTS_PER_CONCURRENCY" | tee -a "$RESULT_FILE"
echo "  Tokenizer Path: $TOKENIZER_PATH" | tee -a "$RESULT_FILE"
echo "  Log Folder: $LOG_FOLDER" | tee -a "$RESULT_FILE"

# Main test execution loop
for qps in "${QPS_LEVELS[@]}"; do
    total_prompts=$(awk -v qps="$qps" -v p="$PROMPTS_PER_QPS" 'BEGIN { printf "%.0f", qps * p }')
    benchmark_log="$LOG_FOLDER/result-${BACKEND}-${CARD}-${INPUT_LENGTH}-${OUTPUT_LENGTH}-${CURRENT_DATE}-${qps}.txt"

    echo -e "\n=== Starting test - QPS: $qps, Total requests: $total_prompts ===" | tee -a "$RESULT_FILE"

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
        --request-rate $qps \
        --num-prompts $total_prompts 2>&1 | tee -a "$benchmark_log" "$RESULT_FILE"

    # Verify command execution status
    if [ $? -ne 0 ]; then
        echo "Warning: Test failed at qps level $qps" | tee -a "$RESULT_FILE"
    fi

    echo "Completed test - QPS: $qps, Total requests: $total_prompts" | tee -a "$RESULT_FILE"
    echo "Results saved to: $RESULT_FILE" | tee -a "$RESULT_FILE"

    # System cooldown period
    echo "Initiating ${COOLDOWN_PERIOD} second cooldown period..." | tee -a "$RESULT_FILE"
    sleep "$COOLDOWN_PERIOD"
done

echo -e "\nAll benchmark tests completed successfully" | tee -a "$RESULT_FILE"

#bash auto_benchmark_qps.sh \
#    --server-ip 0.0.0.0 \
#    --server-port 10001 \
#    --input-length 2048 \
#    --output-length 2048 \
#    --backend vllm \
#    --card H100 \
#    --qps-levels 0.1,0.2,1,2,3 \
#    --prompts-per-qps 600 \
#    --tokenizer-path /home/Llama-3.1-8B-Instruct \
#    --model-name base_model \
#    --log-folder auto_benchmark