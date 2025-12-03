#!/bin/bash

# Benchmark Script for Qwen3-8B Performance Analysis - 4 GPU Configuration
# CS 431/531 Performance Project - llama.cpp benchmarking

# COMMAND to acquire Nodes: salloc --cpus-per-task 64 --mem 200G --gres=gpu:4 --time=1:00:00

# Configuration
MODEL_PATH=~/models/Qwen3-8B/qwen3-8b-q5_k_m.gguf
LLAMA_BENCH=~/llama.cpp/build/bin/llama-bench
THREADS=64
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR=~/perf-analysis-modeling-project/measurements/aaron
OUTPUT_FILE=${OUTPUT_DIR}/benchmark_results_4gpu_${TIMESTAMP}.md

# Colors for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Error handling function
handle_error() {
    local test_name=$1
    local error_msg=$2
    echo -e "${RED}ERROR in $test_name${NC}" | tee -a $OUTPUT_FILE
    echo "" >> $OUTPUT_FILE
    echo "### ❌ Error Encountered" >> $OUTPUT_FILE
    echo '```' >> $OUTPUT_FILE
    echo "$error_msg" >> $OUTPUT_FILE
    echo '```' >> $OUTPUT_FILE
    echo "" >> $OUTPUT_FILE
    echo -e "${YELLOW}Continuing with remaining tests...${NC}"
    echo "" >> $OUTPUT_FILE
}

# Check if files exist
check_prerequisites() {
    local errors=0
    
    if [ ! -f "$MODEL_PATH" ]; then
        echo -e "${RED}ERROR: Model file not found at $MODEL_PATH${NC}"
        errors=1
    fi
    
    if [ ! -f "$LLAMA_BENCH" ]; then
        echo -e "${RED}ERROR: llama-bench not found at $LLAMA_BENCH${NC}"
        errors=1
    fi
    
    # Check if we have 4 GPUs
    GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
    if [ "$GPU_COUNT" -lt 4 ]; then
        echo -e "${YELLOW}WARNING: Only $GPU_COUNT GPU(s) detected. This script is designed for 4 GPUs.${NC}"
        echo -e "${YELLOW}Some tests may fail or not run optimally.${NC}"
    fi
    
    if [ $errors -eq 1 ]; then
        echo -e "${RED}Prerequisites check failed. Exiting.${NC}"
        exit 1
    fi
}

echo -e "${BLUE}=== Qwen3-8B 4-GPU Benchmark Suite ===${NC}"

# Create output directory if it doesn't exist
echo -e "${GREEN}Creating output directory...${NC}"
mkdir -p $OUTPUT_DIR

echo "Output will be saved to: $OUTPUT_FILE"
echo ""

# Check prerequisites
echo -e "${GREEN}Checking prerequisites...${NC}"
check_prerequisites

# Load required modules
echo -e "${GREEN}Loading required modules...${NC}"
if ! module load cuda 2>/dev/null; then
    echo -e "${YELLOW}Warning: Could not load cuda module${NC}"
fi
if ! module load openblas 2>/dev/null; then
    echo -e "${YELLOW}Warning: Could not load openblas module${NC}"
fi

# Create markdown header
cat > $OUTPUT_FILE << 'HEADER'
# Qwen3-8B Performance Benchmark Results - 4 GPU Configuration

## Test Configuration

**Model:** Qwen3-8B (Q5_K_M quantization)  
**Hardware:** ORCA Supercluster  
**Framework:** llama.cpp  
HEADER

echo "**Date:** $(date)" >> $OUTPUT_FILE
echo "**Node:** $(hostname)" >> $OUTPUT_FILE
echo "**Nodes Used:** 1" >> $OUTPUT_FILE
echo "**GPUs per Node:** 4" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

# System Information
echo -e "${GREEN}Collecting system information...${NC}"
echo "## System Information" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
echo '```' >> $OUTPUT_FILE
echo "CPU Info:" >> $OUTPUT_FILE
lscpu | grep -E "Model name|CPU\(s\)|Thread|Core" >> $OUTPUT_FILE 2>&1 || echo "Could not retrieve CPU info" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
echo "GPU Info:" >> $OUTPUT_FILE
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv >> $OUTPUT_FILE 2>&1 || echo "Could not retrieve GPU info" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
echo "CUDA Toolkit Version:" >> $OUTPUT_FILE
nvcc --version | grep "release" >> $OUTPUT_FILE 2>&1 || echo "Could not retrieve CUDA version" >> $OUTPUT_FILE
echo '```' >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

# Benchmark configurations
echo "## Benchmark Configurations" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
echo "| Configuration | GPUs Used | Tensor Split | CPU Threads | Description |" >> $OUTPUT_FILE
echo "|---------------|-----------|--------------|-------------|-------------|" >> $OUTPUT_FILE
echo "| Single GPU | 1 | N/A | 64 | Baseline: All layers on GPU 0 |" >> $OUTPUT_FILE
echo "| Dual GPU | 2 | 8,8,0,0 | 64 | Split across GPU 0 and 1 |" >> $OUTPUT_FILE
echo "| Quad GPU Balanced | 4 | 4,4,4,4 | 64 | Evenly distributed across all 4 GPUs |" >> $OUTPUT_FILE
echo "| Quad GPU Custom | 4 | 5,5,3,3 | 64 | Weighted distribution (testing variation) |" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "---" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

# Test 1: Single GPU (Baseline)
echo -e "${GREEN}[1/4] Running Single GPU benchmark (baseline)...${NC}"
echo "## Test 1: Single GPU (Baseline)" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
echo "**Configuration:** All layers on GPU 0" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
echo '```' >> $OUTPUT_FILE

GPU1_OUTPUT=$(CUDA_VISIBLE_DEVICES=0 timeout 600 $LLAMA_BENCH -m $MODEL_PATH -ngl 99 -t $THREADS 2>&1)
GPU1_EXIT_CODE=$?

if [ $GPU1_EXIT_CODE -eq 0 ] && echo "$GPU1_OUTPUT" | grep -q "t/s"; then
    echo "$GPU1_OUTPUT" | tee -a $OUTPUT_FILE
    echo -e "${GREEN}✓ Test 1 completed successfully${NC}"
else
    echo "$GPU1_OUTPUT" >> $OUTPUT_FILE
    handle_error "Test 1: Single GPU" "Benchmark failed or timed out after 10 minutes (exit code: $GPU1_EXIT_CODE)"
fi
echo '```' >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
sleep 2

# Test 2: Dual GPU
echo -e "${GREEN}[2/4] Running Dual GPU benchmark...${NC}"
echo "## Test 2: Dual GPU (2 GPUs)" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
echo "**Configuration:** Tensor split 8,8,0,0 (using GPU 0 and 1)" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
echo '```' >> $OUTPUT_FILE

GPU2_OUTPUT=$(timeout 600 $LLAMA_BENCH -m $MODEL_PATH -ngl 99 -ts 8,8,0,0 -t $THREADS 2>&1)
GPU2_EXIT_CODE=$?

if [ $GPU2_EXIT_CODE -eq 0 ] && echo "$GPU2_OUTPUT" | grep -q "t/s"; then
    echo "$GPU2_OUTPUT" | tee -a $OUTPUT_FILE
    echo -e "${GREEN}✓ Test 2 completed successfully${NC}"
else
    echo "$GPU2_OUTPUT" >> $OUTPUT_FILE
    handle_error "Test 2: Dual GPU" "Benchmark failed or timed out after 10 minutes (exit code: $GPU2_EXIT_CODE)"
fi
echo '```' >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
sleep 2

# Test 3: Quad GPU Balanced
echo -e "${GREEN}[3/4] Running Quad GPU benchmark (balanced)...${NC}"
echo "## Test 3: Quad GPU - Balanced Distribution" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
echo "**Configuration:** Tensor split 4,4,4,4 (evenly distributed)" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
echo '```' >> $OUTPUT_FILE

GPU4_OUTPUT=$(timeout 600 $LLAMA_BENCH -m $MODEL_PATH -ngl 99 -ts 4,4,4,4 -t $THREADS 2>&1)
GPU4_EXIT_CODE=$?

if [ $GPU4_EXIT_CODE -eq 0 ] && echo "$GPU4_OUTPUT" | grep -q "t/s"; then
    echo "$GPU4_OUTPUT" | tee -a $OUTPUT_FILE
    echo -e "${GREEN}✓ Test 3 completed successfully${NC}"
else
    echo "$GPU4_OUTPUT" >> $OUTPUT_FILE
    handle_error "Test 3: Quad GPU Balanced" "Benchmark failed or timed out after 10 minutes (exit code: $GPU4_EXIT_CODE)"
fi
echo '```' >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
sleep 2

# Test 4: Quad GPU Custom Split
echo -e "${GREEN}[4/4] Running Quad GPU benchmark (custom split)...${NC}"
echo "## Test 4: Quad GPU - Custom Distribution" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
echo "**Configuration:** Tensor split 5,5,3,3 (weighted distribution)" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
echo '```' >> $OUTPUT_FILE

GPU4C_OUTPUT=$(timeout 600 $LLAMA_BENCH -m $MODEL_PATH -ngl 99 -ts 5,5,3,3 -t $THREADS 2>&1)
GPU4C_EXIT_CODE=$?

if [ $GPU4C_EXIT_CODE -eq 0 ] && echo "$GPU4C_OUTPUT" | grep -q "t/s"; then
    echo "$GPU4C_OUTPUT" | tee -a $OUTPUT_FILE
    echo -e "${GREEN}✓ Test 4 completed successfully${NC}"
else
    echo "$GPU4C_OUTPUT" >> $OUTPUT_FILE
    handle_error "Test 4: Quad GPU Custom" "Benchmark failed or timed out after 10 minutes (exit code: $GPU4C_EXIT_CODE)"
fi
echo '```' >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

# Summary section
echo "## Performance Summary" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
echo "### Multi-GPU Scaling Analysis" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
echo "| Configuration | GPUs | Prompt Processing (pp512) | Text Generation (tg128) | Speedup vs 1 GPU (pp) | Speedup vs 1 GPU (tg) |" >> $OUTPUT_FILE
echo "|---------------|------|---------------------------|-------------------------|-----------------------|-----------------------|" >> $OUTPUT_FILE
echo "| Single GPU | 1 | - | - | 1.00x | 1.00x |" >> $OUTPUT_FILE
echo "| Dual GPU | 2 | - | - | - | - |" >> $OUTPUT_FILE
echo "| Quad GPU (Balanced) | 4 | - | - | - | - |" >> $OUTPUT_FILE
echo "| Quad GPU (Custom) | 4 | - | - | - | - |" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
echo "*Note: Fill in actual values from results above and calculate speedups*" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "## Observations" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
echo "### Prompt Processing (pp512)" >> $OUTPUT_FILE
echo "- **Single GPU Performance:** " >> $OUTPUT_FILE
echo "- **Dual GPU Scaling:** " >> $OUTPUT_FILE
echo "- **Quad GPU Scaling:** " >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
echo "### Text Generation (tg128)" >> $OUTPUT_FILE
echo "- **Single GPU Performance:** " >> $OUTPUT_FILE
echo "- **Dual GPU Scaling:** " >> $OUTPUT_FILE
echo "- **Quad GPU Scaling:** " >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
echo "### Multi-GPU Efficiency" >> $OUTPUT_FILE
echo "- **Linear Scaling?:** " >> $OUTPUT_FILE
echo "- **Communication Overhead:** " >> $OUTPUT_FILE
echo "- **Optimal Configuration:** " >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "## Build Configuration" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
echo '```' >> $OUTPUT_FILE
cat ~/llama.cpp/build/CMakeCache.txt | grep -E "GGML_CUDA|GGML_BLAS|CMAKE_BUILD_TYPE" | grep -v "ADVANCED" >> $OUTPUT_FILE 2>&1 || echo "Could not retrieve build configuration" >> $OUTPUT_FILE
echo '```' >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "---" >> $OUTPUT_FILE
echo "*Generated by automated 4-GPU benchmark script*" >> $OUTPUT_FILE
echo "*Stored in: $OUTPUT_FILE*" >> $OUTPUT_FILE

echo ""
echo -e "${BLUE}=== 4-GPU Benchmark Complete! ===${NC}"
echo -e "${GREEN}Results saved to: $OUTPUT_FILE${NC}"
echo ""
echo "You can view the results with:"
echo "  cat $OUTPUT_FILE"
echo "  or"
echo "  less $OUTPUT_FILE"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "1. Compare these results with your 1-GPU benchmarks"
echo "2. Calculate scaling efficiency (speedup / number of GPUs)"
echo "3. Analyze which workload benefits most from multi-GPU"
