# perf-analysis-modeling-project

## lama-cpp deployment notes on `orca` cluster (with CUDA and OpenBLAS)

### llama-cpp build

1. It is built in `/scratch/mmarkoc-pdx_performance` directory
2. Log into one of the compute nodes for faster build
	* `salloc --cpus-per-task 64 --mem 200G --gres=gpu:1`
3. Load the openblas,cuda and cmake module
	* `module load cuda`
	* `module load openblas`
	* `module load cmake` 
4. `git clone https://github.com/ggml-org/llama.cpp.git`
5. `cd llama.cpp`
6.  Build
	* `cmake -B build -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS -DGGML_CUDA=ON`
		* to build CPU only, run `cmake -B build -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS`
	* `cmake --build build --config Release`
	* All of the binaries are in the `build/bin`
###  Getting a model

* `/build/bin/llama-server -hf MaziyarPanahi/VibeThinker-1.5B-GGUF:Q4_K_M`
* Optional
	* copy or move model to `/scratch/mmarkoc-pdx_performance/models`

### Running a benchmark

* `/build/bin/llama-bench -m <path_to_the_model>`

* Note: llama.cpp is built with Qwen 3: https://huggingface.co/Qwen/Qwen3-8B?inference_provider=nscale


