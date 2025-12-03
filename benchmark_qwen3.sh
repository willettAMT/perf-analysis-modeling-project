#!/bin/bash

# Benchmark Script for Qwen3-8B Performance Analysis
# CS 431/531 Performance Project - llama.cpp benchmarking

# COMMAND to acquire Nodes: salloc --cpus-per-task 64 --mem 200G --gres=gpu:1 --time=1:00:00

# Configuration
MODEL_PATH=~/models/Qwen3-8B/qwen3-8b-q5_k_m.gguf
LLAMA_BENCH=~/llama.cpp/build/bin/llama-bench
THREADS=64
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR=~/perf-analysis-modeling-project/measurements/aaron
OUTPUT_FILE=${OUTPUT_DIR}/benchmark_results_${TIMESTAMP}.md

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
    
    if [ $errors -eq 1 ]; then
        echo -e "${RED}Prerequisites check failed. Exiting.${NC}"
        exit 1
    fi
}

echo -e "${BLUE}=== Qwen3-8B Benchmark Suite ===${NC}"

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
# Qwen3-8B Performance Benchmark Results

## Test Configuration

**Model:** Qwen3-8B (Q5_K_M quantization)  
**Hardware:** ORCA Supercluster  
**Framework:** llama.cpp  
HEADER

echo "**Date:** $(date)" >> $OUTPUT_FILE
echo "**Node:** $(hostname)" >> $OUTPUT_FILE
echo "**Nodes Used:** 1" >> $OUTPUT_FILE
echo "**GPUs per Node:** 1" >> $OUTPUT_FILE
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
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader >> $OUTPUT_FILE 2>&1 || echo "Could not retrieve GPU info" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
echo "CUDA Toolkit Version:" >> $OUTPUT_FILE
nvcc --version | grep "release" >> $OUTPUT_FILE 2>&1 || echo "Could not retrieve CUDA version" >> $OUTPUT_FILE
echo '```' >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

# Benchmark configurations
echo "## Benchmark Configurations" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
echo "| Configuration | Layers on GPU | CPU Threads | Description |" >> $OUTPUT_FILE
echo "|---------------|---------------|-------------|-------------|" >> $OUTPUT_FILE
echo "| CPU-Only | 0 | 64 | Pure CPU processing with OpenBLAS |" >> $OUTPUT_FILE
echo "| GPU Partial | 10 | 64 | Hybrid: 10 layers on GPU, rest on CPU |" >> $OUTPUT_FILE
echo "| GPU Full | 99 (all) | 64 | All layers offloaded to GPU |" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "---" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

# Test 1: CPU-Only (64 threads)
echo -e "${GREEN}[1/3] Running CPU-Only benchmark (64 threads)...${NC}"
echo "## Test 1: CPU-Only (64 threads)" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
echo '```' >> $OUTPUT_FILE

# Run CPU-only test and capture output
CPU_OUTPUT=$(CUDA_VISIBLE_DEVICES="" timeout 600 $LLAMA_BENCH -m $MODEL_PATH -t $THREADS 2>&1)
CPU_EXIT_CODE=$?

# Check if it completed successfully (look for benchmark results in output)
if [ $CPU_EXIT_CODE -eq 0 ] && echo "$CPU_OUTPUT" | grep -q "t/s"; then
    echo "$CPU_OUTPUT" | tee -a $OUTPUT_FILE
    echo -e "${GREEN}✓ Test 1 completed successfully${NC}"
else
    echo "$CPU_OUTPUT" >> $OUTPUT_FILE
    handle_error "Test 1: CPU-Only" "Benchmark failed or timed out after 10 minutes (exit code: $CPU_EXIT_CODE)"
fi
echo '```' >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
sleep 2

# Test 2: GPU with 10 layers offloaded
echo -e "${GREEN}[2/3] Running GPU benchmark with 10 layers offloaded...${NC}"
echo "## Test 2: GPU Partial Offloading (10 layers)" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
echo '```' >> $OUTPUT_FILE

GPU10_OUTPUT=$(timeout 600 $LLAMA_BENCH -m $MODEL_PATH -ngl 10 -t $THREADS 2>&1)
GPU10_EXIT_CODE=$?

if [ $GPU10_EXIT_CODE -eq 0 ] && echo "$GPU10_OUTPUT" | grep -q "t/s"; then
    echo "$GPU10_OUTPUT" | tee -a $OUTPUT_FILE
    echo -e "${GREEN}✓ Test 2 completed successfully${NC}"
else
    echo "$GPU10_OUTPUT" >> $OUTPUT_FILE
    handle_error "Test 2: GPU Partial" "Benchmark failed or timed out after 10 minutes (exit code: $GPU10_EXIT_CODE)"
fi
echo '```' >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
sleep 2

# Test 3: GPU with all layers offloaded
echo -e "${GREEN}[3/3] Running GPU benchmark with full offloading (all layers)...${NC}"
echo "## Test 3: GPU Full Offloading (all layers)" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
echo '```' >> $OUTPUT_FILE

GPU99_OUTPUT=$(timeout 600 $LLAMA_BENCH -m $MODEL_PATH -ngl 99 -t $THREADS 2>&1)
GPU99_EXIT_CODE=$?

if [ $GPU99_EXIT_CODE -eq 0 ] && echo "$GPU99_OUTPUT" | grep -q "t/s"; then
    echo "$GPU99_OUTPUT" | tee -a $OUTPUT_FILE
    echo -e "${GREEN}✓ Test 3 completed successfully${NC}"
else
    echo "$GPU99_OUTPUT" >> $OUTPUT_FILE
    handle_error "Test 3: GPU Full" "Benchmark failed or timed out after 10 minutes (exit code: $GPU99_EXIT_CODE)"
fi
echo '```' >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

# Summary section
echo "## Performance Summary" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
echo "### Tokens per Second Comparison" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
echo "| Configuration | Prompt Processing (pp512) | Text Generation (tg128) | Speedup vs CPU (pp) | Speedup vs CPU (tg) |" >> $OUTPUT_FILE
echo "|---------------|---------------------------|-------------------------|---------------------|---------------------|" >> $OUTPUT_FILE

# Extract results (this is a simplified extraction - you may need to adjust)
echo "*Note: Fill in speedup calculations manually from results above*" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "## Observations" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
echo "- **Prompt Processing:** " >> $OUTPUT_FILE
echo "- **Text Generation:** " >> $OUTPUT_FILE
echo "- **Memory Usage:** " >> $OUTPUT_FILE
echo "- **Optimal Configuration:** " >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "## Build Configuration" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
echo '```' >> $OUTPUT_FILE
cat ~/llama.cpp/build/CMakeCache.txt | grep -E "GGML_CUDA|GGML_BLAS|CMAKE_BUILD_TYPE" | grep -v "ADVANCED" >> $OUTPUT_FILE 2>&1 || echo "Could not retrieve build configuration" >> $OUTPUT_FILE
echo '```' >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "---" >> $OUTPUT_FILE
echo "*Generated by automated benchmark script*" >> $OUTPUT_FILE
echo "*Stored in: $OUTPUT_FILE*" >> $OUTPUT_FILE

echo ""
echo -e "${BLUE}=== Benchmark Complete! ===${NC}"
echo -e "${GREEN}Results saved to: $OUTPUT_FILE${NC}"
echo ""
echo "You can view the results with:"
echo "  cat $OUTPUT_FILE"
echo "  or"
echo "  less $OUTPUT_FILE"
