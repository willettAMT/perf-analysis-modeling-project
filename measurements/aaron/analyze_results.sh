#!/usr/bin/env python3
"""
Dynamic Benchmark Analysis Script
Automatically parses benchmark markdown files and generates analysis
"""

import re
import os
from pathlib import Path

# Configuration
BENCHMARK_DIR = Path.home() / "perf-analysis-modeling-project/measurements/aaron"

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
    
    # Extract benchmark results - look for the table rows
    # Pattern: | qwen3 8B Q5_K - Medium | size | params | backend | threads | test | t/s |
    pattern = r'\|\s*qwen3 8B.*?\|\s*[\d.]+\s*GiB\s*\|\s*[\d.]+\s*B\s*\|\s*[\w,]+\s*\|\s*(\d+)\s*\|(?:\s*[\d.]+\s*\|)?\s*(pp\d+|tg\d+)\s*\|\s*([\d.]+)'
    
    matches = re.finditer(pattern, content)
    for match in matches:
        threads = int(match.group(1))
        test_type = match.group(2)
        tokens_per_sec = float(match.group(3))
        
        results['tests'].append({
            'threads': threads,
            'test_type': test_type,
            'tokens_per_sec': tokens_per_sec
        })
    
    return results

def aggregate_tests(tests):
    """Aggregate test results by type"""
    aggregated = {}
    for test in tests:
        test_type = test['test_type']
        if test_type not in aggregated:
            aggregated[test_type] = []
        aggregated[test_type].append(test['tokens_per_sec'])
    
    # Calculate averages
    averages = {}
    for test_type, values in aggregated.items():
        averages[test_type] = sum(values) / len(values)
    
    return averages

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
            
            aggregated = aggregate_tests(result['tests'])
            
            print("\nPerformance Metrics:")
            print(f"  Prompt Processing (pp512): {aggregated.get('pp512', 0):.2f} t/s")
            print(f"  Text Generation (tg128):   {aggregated.get('tg128', 0):.2f} t/s")
            
            # Calculate speedups if we have CPU baseline
            if 'CPU' in result['filename'] or aggregated.get('pp512', 0) < 100:
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
            
            aggregated = aggregate_tests(result['tests'])
            
            print("\nPerformance Metrics:")
            print(f"  Prompt Processing (pp512): {aggregated.get('pp512', 0):.2f} t/s")
            print(f"  Text Generation (tg128):   {aggregated.get('tg128', 0):.2f} t/s")

def generate_comparison_table(results_list):
    """Generate comparison table across all configurations"""
    print("\n" + "="*70)
    print("COMPREHENSIVE COMPARISON TABLE")
    print("="*70)
    print()
    print("| Node      | GPU Type  | GPUs | Prompt (pp512) | Generation (tg128) |")
    print("|-----------|-----------|------|----------------|-------------------|")
    
    for result in results_list:
        aggregated = aggregate_tests(result['tests'])
        node = result['node'] or 'Unknown'
        gpu_type = result['gpu_type'] or 'CPU-Only'
        gpu_count = result['gpu_count'] or 0
        pp512 = aggregated.get('pp512', 0)
        tg128 = aggregated.get('tg128', 0)
        
        print(f"| {node:9s} | {gpu_type:9s} | {gpu_count:4d} | {pp512:14.2f} | {tg128:17.2f} |")

def calculate_speedups(results_list):
    """Calculate speedups comparing GPU to CPU"""
    print("\n" + "="*70)
    print("SPEEDUP ANALYSIS")
    print("="*70)
    
    # Find CPU baseline
    cpu_baseline = None
    for result in results_list:
        aggregated = aggregate_tests(result['tests'])
        if aggregated.get('pp512', 0) < 100:  # Likely CPU-only
            cpu_baseline = aggregated
            break
    
    if not cpu_baseline:
        print("\nNo CPU baseline found for speedup calculation")
        return
    
    print(f"\nCPU Baseline:")
    print(f"  Prompt Processing: {cpu_baseline.get('pp512', 0):.2f} t/s")
    print(f"  Text Generation:   {cpu_baseline.get('tg128', 0):.2f} t/s")
    print("\nGPU Speedups:")
    print()
    print("| Configuration | Prompt Speedup | Generation Speedup |")
    print("|---------------|----------------|-------------------|")
    
    for result in results_list:
        aggregated = aggregate_tests(result['tests'])
        pp512 = aggregated.get('pp512', 0)
        tg128 = aggregated.get('tg128', 0)
        
        if pp512 > 100:  # GPU configuration
            pp_speedup = pp512 / cpu_baseline.get('pp512', 1)
            tg_speedup = tg128 / cpu_baseline.get('tg128', 1)
            
            config_name = f"{result['node']} ({result['gpu_type']})"
            print(f"| {config_name:13s} | {pp_speedup:14.2f}x | {tg_speedup:17.2f}x |")

def generate_key_findings(results_list):
    """Generate key findings summary"""
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    
    # Find best and worst performers
    best_pp = max(results_list, key=lambda r: aggregate_tests(r['tests']).get('pp512', 0))
    best_tg = max(results_list, key=lambda r: aggregate_tests(r['tests']).get('tg128', 0))
    
    best_pp_agg = aggregate_tests(best_pp['tests'])
    best_tg_agg = aggregate_tests(best_tg['tests'])
    
    print(f"\n1. Best Prompt Processing Performance:")
    print(f"   {best_pp['node']} - {best_pp['gpu_type']}")
    print(f"   {best_pp_agg.get('pp512', 0):.2f} t/s")
    
    print(f"\n2. Best Text Generation Performance:")
    print(f"   {best_tg['node']} - {best_tg['gpu_type']}")
    print(f"   {best_tg_agg.get('tg128', 0):.2f} t/s")
    
    # Multi-GPU analysis
    multi_gpu_results = [r for r in results_list if r['gpu_count'] and r['gpu_count'] > 1]
    single_gpu_results = [r for r in results_list if r['gpu_count'] == 1 and aggregate_tests(r['tests']).get('pp512', 0) > 1000]
    
    if multi_gpu_results and single_gpu_results:
        print(f"\n3. Multi-GPU Scaling:")
        single_avg = sum(aggregate_tests(r['tests']).get('pp512', 0) for r in single_gpu_results) / len(single_gpu_results)
        multi_avg = sum(aggregate_tests(r['tests']).get('pp512', 0) for r in multi_gpu_results) / len(multi_gpu_results)
        
        if multi_avg < single_avg:
            print(f"   ⚠ Multi-GPU shows NEGATIVE scaling")
            print(f"   Single GPU avg: {single_avg:.2f} t/s")
            print(f"   Multi GPU avg:  {multi_avg:.2f} t/s")
            print(f"   Performance loss: {((single_avg - multi_avg) / single_avg * 100):.1f}%")
        else:
            print(f"   ✓ Multi-GPU shows positive scaling")

def main():
    print("="*70)
    print("LLAMA.CPP BENCHMARK ANALYSIS")
    print("="*70)
    print(f"\nScanning directory: {BENCHMARK_DIR}")
    
    # Find all benchmark markdown files
    benchmark_files = list(BENCHMARK_DIR.glob("benchmark_results*.md"))
    
    if not benchmark_files:
        print(f"\n❌ No benchmark files found in {BENCHMARK_DIR}")
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

if __name__ == "__main__":
    main()
