#!/usr/bin/env python3
"""
Dynamic Benchmark Analysis Script
Automatically parses benchmark markdown files and generates analysis
"""

import re
import os
import sys
from pathlib import Path
from datetime import datetime

# Configuration
BENCHMARK_DIR = Path.home() / "perf-analysis-modeling-project/measurements/aaron"
OUTPUT_FILE = BENCHMARK_DIR / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

class Tee:
    """Write to both stdout and file simultaneously"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

def parse_benchmark_file(filepath):
    """Parse a benchmark markdown file and extract performance data"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    results = {
        'filepath': str(filepath),
        'filename': filepath.name,
        'node': None,
        'gpu_type': None,
        'gpu_count': None,
        'tests': []
    }
    
    # Extract node information
    node_match = re.search(r'\*\*Node:\*\* (\w+)', content)
    if node_match:
        results['node'] = node_match.group(1)
    
    # Extract GPU information
    gpu_match = re.search(r'(\d+), (NVIDIA \w+)', content)
    if gpu_match:
        results['gpu_type'] = gpu_match.group(2)
    
    # Extract GPU count
    gpu_count_match = re.search(r'\*\*GPUs per Node:\*\* (\d+)', content)
    if gpu_count_match:
        results['gpu_count'] = int(gpu_count_match.group(1))
    
    # Check if this is a CPU-only test
    is_cpu_only = 'failed to initialize CUDA' in content or 'CPU-Only' in content
    
    # Extract benchmark results
    pattern = r'\|\s*qwen3 8B.*?\|\s*[\d.]+\s*GiB\s*\|\s*[\d.]+\s*B\s*\|\s*[\w,]+\s*\|\s*(\d+)\s*\|(?:\s*[\d.]+\s*\|)?\s*(pp\d+|tg\d+)\s*\|\s*([\d.]+)'
    
    matches = re.finditer(pattern, content)
    for match in matches:
        threads = int(match.group(1))
        test_type = match.group(2)
        tokens_per_sec = float(match.group(3))
        
        results['tests'].append({
            'threads': threads,
            'test_type': test_type,
            'tokens_per_sec': tokens_per_sec,
            'is_cpu_only': is_cpu_only
        })
    
    return results

def aggregate_tests(tests):
    """Aggregate test results by type"""
    aggregated = {}
    is_cpu = False
    
    for test in tests:
        test_type = test['test_type']
        if test_type not in aggregated:
            aggregated[test_type] = []
        aggregated[test_type].append(test['tokens_per_sec'])
        if test.get('is_cpu_only'):
            is_cpu = True
    
    # Calculate averages
    averages = {}
    for test_type, values in aggregated.items():
        averages[test_type] = sum(values) / len(values)
    
    return averages, is_cpu

def analyze_single_gpu_results(results_list):
    """Analyze single GPU configuration results"""
    print("\n" + "="*70)
    print("SINGLE GPU CONFIGURATION ANALYSIS")
    print("="*70)
    
    for result in results_list:
        if result['gpu_count'] == 1:
            print(f"\nNode: {result['node']}")
            print(f"GPU: {result['gpu_type']}")
            print(f"File: {result['filename']}")
            
            aggregated, is_cpu = aggregate_tests(result['tests'])
            
            print("\nPerformance Metrics:")
            print(f"  Prompt Processing (pp512): {aggregated.get('pp512', 0):.2f} t/s")
            print(f"  Text Generation (tg128):   {aggregated.get('tg128', 0):.2f} t/s")
            
            if is_cpu:
                print("  → CPU-Only Baseline")
            else:
                print("  → GPU-Accelerated")

def analyze_multi_gpu_results(results_list):
    """Analyze multi-GPU configuration results"""
    print("\n" + "="*70)
    print("MULTI-GPU CONFIGURATION ANALYSIS")
    print("="*70)
    
    for result in results_list:
        if result['gpu_count'] and result['gpu_count'] > 1:
            print(f"\nNode: {result['node']}")
            print(f"GPU: {result['gpu_type']} x {result['gpu_count']}")
            print(f"File: {result['filename']}")
            
            aggregated, _ = aggregate_tests(result['tests'])
            
            print("\nPerformance Metrics:")
            print(f"  Prompt Processing (pp512): {aggregated.get('pp512', 0):.2f} t/s")
            print(f"  Text Generation (tg128):   {aggregated.get('tg128', 0):.2f} t/s")

def generate_comparison_table(results_list):
    """Generate comparison table across all configurations"""
    print("\n" + "="*70)
    print("COMPREHENSIVE COMPARISON TABLE")
    print("="*70)
    print()
    print("| Node      | GPU Type  | GPUs | Config      | Prompt (pp512) | Generation (tg128) |")
    print("|-----------|-----------|------|-------------|----------------|-------------------|")
    
    for result in results_list:
        aggregated, is_cpu = aggregate_tests(result['tests'])
        node = result['node'] or 'Unknown'
        gpu_type = result['gpu_type'] or 'CPU'
        gpu_count = result['gpu_count'] or 0
        config = "CPU-Only" if is_cpu else "GPU-Full"
        pp512 = aggregated.get('pp512', 0)
        tg128 = aggregated.get('tg128', 0)
        
        print(f"| {node:9s} | {gpu_type:9s} | {gpu_count:4d} | {config:11s} | {pp512:14.2f} | {tg128:17.2f} |")

def calculate_speedups(results_list):
    """Calculate speedups comparing GPU to CPU"""
    print("\n" + "="*70)
    print("SPEEDUP ANALYSIS")
    print("="*70)
    
    # Find CPU baseline
    cpu_baseline = None
    cpu_result = None
    
    for result in results_list:
        aggregated, is_cpu = aggregate_tests(result['tests'])
        if is_cpu:
            cpu_baseline = aggregated
            cpu_result = result
            break
    
    if not cpu_baseline:
        print("\nNo CPU baseline found for speedup calculation")
        return
    
    print(f"\nCPU Baseline ({cpu_result['node']}):")
    print(f"  Prompt Processing: {cpu_baseline.get('pp512', 0):.2f} t/s")
    print(f"  Text Generation:   {cpu_baseline.get('tg128', 0):.2f} t/s")
    print("\nGPU Speedups:")
    print()
    print("| Node      | GPU Type     | GPUs | Prompt Speedup | Generation Speedup |")
    print("|-----------|--------------|------|----------------|-------------------|")
    
    for result in results_list:
        aggregated, is_cpu = aggregate_tests(result['tests'])
        
        if not is_cpu:  # GPU configuration
            pp512 = aggregated.get('pp512', 0)
            tg128 = aggregated.get('tg128', 0)
            
            pp_speedup = pp512 / cpu_baseline.get('pp512', 1)
            tg_speedup = tg128 / cpu_baseline.get('tg128', 1)
            
            node = result['node'] or 'Unknown'
            gpu_type = result['gpu_type'] or 'Unknown'
            gpu_count = result['gpu_count'] or 1
            
            print(f"| {node:9s} | {gpu_type:12s} | {gpu_count:4d} | {pp_speedup:14.2f}x | {tg_speedup:17.2f}x |")

def generate_key_findings(results_list):
    """Generate key findings summary"""
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    
    # Find best performers (excluding CPU)
    gpu_results = []
    for r in results_list:
        agg, is_cpu = aggregate_tests(r['tests'])
        if not is_cpu:
            gpu_results.append((r, agg))
    
    if not gpu_results:
        print("\nNo GPU results found")
        return
    
    best_pp = max(gpu_results, key=lambda x: x[1].get('pp512', 0))
    best_tg = max(gpu_results, key=lambda x: x[1].get('tg128', 0))
    
    print(f"\n1. Best Prompt Processing Performance:")
    print(f"   {best_pp[0]['node']} - {best_pp[0]['gpu_type']}")
    print(f"   {best_pp[1].get('pp512', 0):.2f} t/s")
    
    print(f"\n2. Best Text Generation Performance:")
    print(f"   {best_tg[0]['node']} - {best_tg[0]['gpu_type']}")
    print(f"   {best_tg[1].get('tg128', 0):.2f} t/s")
    
    # Multi-GPU analysis
    multi_gpu_results = [(r, agg) for r, agg in gpu_results if r['gpu_count'] and r['gpu_count'] > 1]
    single_gpu_results = [(r, agg) for r, agg in gpu_results if r['gpu_count'] == 1]
    
    if multi_gpu_results and single_gpu_results:
        print(f"\n3. Multi-GPU Scaling:")
        single_avg = sum(agg.get('pp512', 0) for _, agg in single_gpu_results) / len(single_gpu_results)
        multi_avg = sum(agg.get('pp512', 0) for _, agg in multi_gpu_results) / len(multi_gpu_results)
        
        if multi_avg < single_avg:
            print(f"   ⚠ Multi-GPU shows NEGATIVE scaling")
            print(f"   Single GPU avg: {single_avg:.2f} t/s")
            print(f"   Multi GPU avg:  {multi_avg:.2f} t/s")
            print(f"   Performance loss: {((single_avg - multi_avg) / single_avg * 100):.1f}%")
        else:
            print(f"   ✓ Multi-GPU shows positive scaling")
            print(f"   Single GPU avg: {single_avg:.2f} t/s")
            print(f"   Multi GPU avg:  {multi_avg:.2f} t/s")
            print(f"   Performance gain: {((multi_avg - single_avg) / single_avg * 100):.1f}%")
    
    # Hardware comparison
    gpu_types = {}
    for r, agg in gpu_results:
        gpu_type = r['gpu_type']
        if gpu_type not in gpu_types:
            gpu_types[gpu_type] = []
        gpu_types[gpu_type].append(agg.get('pp512', 0))
    
    if len(gpu_types) > 1:
        print(f"\n4. Hardware Comparison:")
        for gpu_type, values in gpu_types.items():
            avg = sum(values) / len(values)
            print(f"   {gpu_type}: {avg:.2f} t/s (avg)")

def main():
    # Redirect output to both terminal and file
    tee = Tee(OUTPUT_FILE)
    sys.stdout = tee
    
    print("="*70)
    print("LLAMA.CPP BENCHMARK ANALYSIS")
    print("="*70)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nScanning directory: {BENCHMARK_DIR}")
    
    # Find all benchmark markdown files
    benchmark_files = list(BENCHMARK_DIR.glob("benchmark_results*.md"))
    
    if not benchmark_files:
        print(f"\n❌ No benchmark files found in {BENCHMARK_DIR}")
        tee.close()
        sys.stdout = tee.terminal
        return
    
    print(f"Found {len(benchmark_files)} benchmark file(s)")
    
    # Parse all files
    results_list = []
    for filepath in benchmark_files:
        print(f"  - {filepath.name}")
        try:
            results = parse_benchmark_file(filepath)
            if results['tests']:
                results_list.append(results)
        except Exception as e:
            print(f"    ⚠ Error parsing {filepath.name}: {e}")
    
    if not results_list:
        print("\n❌ No valid results found")
        tee.close()
        sys.stdout = tee.terminal
        return
    
    # Run analyses
    analyze_single_gpu_results(results_list)
    analyze_multi_gpu_results(results_list)
    generate_comparison_table(results_list)
    calculate_speedups(results_list)
    generate_key_findings(results_list)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\n✅ Analysis saved to: {OUTPUT_FILE}")
    
    # Close file and restore stdout
    tee.close()
    sys.stdout = tee.terminal
    
    print(f"\n✅ Analysis saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
