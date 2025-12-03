# Qwen3-8B Performance Benchmark Results

## Test Configuration

**Model:** Qwen3-8B (Q5_K_M quantization)  
**Hardware:** ORCA Supercluster  
**Framework:** llama.cpp  
**Date:** Tue Dec  2 08:09:51 PM PST 2025
**Node:** login
**Nodes Used:** 1
**GPUs per Node:** 1

## System Information

```
CPU Info:
CPU(s):                               4
On-line CPU(s) list:                  0-3
Model name:                           Intel(R) Xeon(R) Gold 6342 CPU @ 2.80GHz
Thread(s) per core:                   1
Core(s) per socket:                   1
NUMA node0 CPU(s):                    0-3

GPU Info:
./benchmark_qwen3.sh: line 106: nvidia-smi: command not found
Could not retrieve GPU info

CUDA Toolkit Version:
Cuda compilation tools, release 12.9, V12.9.41
```

## Benchmark Configurations

| Configuration | Layers on GPU | CPU Threads | Description |
|---------------|---------------|-------------|-------------|
| CPU-Only | 0 | 64 | Pure CPU processing with OpenBLAS |
| GPU Partial | 10 | 64 | Hybrid: 10 layers on GPU, rest on CPU |
| GPU Full | 99 (all) | 64 | All layers offloaded to GPU |

---

## Test 1: CPU-Only (64 threads)

```
/home/aaronw/llama.cpp/build/bin/llama-bench: error while loading shared libraries: libcuda.so.1: cannot open shared object file: No such file or directory
[0;31mERROR in Test 1: CPU-Only[0m

### ❌ Error Encountered
```
Benchmark failed or timed out after 10 minutes (exit code: 127)
```


```

