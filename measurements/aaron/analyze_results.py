#!/usr/bin/env python3
"""
Robust Benchmark Analysis Script
Uses simpler line-by-line parsing instead of complex regex
"""

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
    """Parse benchmark file line by line"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    results = {
        'filepath': str(filepath),
        'filename': filepath.name,
        'node': None,
        'gpu_type': None,
        'gpu_count': None,
        'configurations': []
    }
    
    current_config = None
    in_results_table = False
    
    for i, line in enumerate(lines):
        # Extract metadata
        if '**Node:**' in line:
            results['node'] = line.split('**Node:**')[1].strip()
        elif '**GPUs per Node:**' in line:
            results['gpu_count'] = int(line.split('**GPUs per Node:**')[1].strip())
        elif 'Device 0:' in line and 'NVIDIA' in line:
            # Extract GPU type from "Device 0: NVIDIA A30, compute..."
            parts = line.split('NVIDIA')[1].split(',')[0].strip()
            results['gpu_type'] = f"NVIDIA {parts}"
        
        # Detect test sections
        if line.startswith('## Test'):
            # Save previous config if exists
            if current_config and (current_config['pp512'] or current_config['tg128']):
                avg_pp = sum(current_config['pp512']) / len(current_config['pp512']) if current_config['pp512'] else 0
                avg_tg = sum(current_config['tg128']) / len(current_config['tg128']) if current_config['tg128'] else 0
                results['configurations'].append({
                    'name': current_config['name'],
                    'is_cpu_only': current_config['is_cpu'],
                    'pp512': avg_pp,
                    'tg128': avg_tg,
                    'test_num': current_config['test_num']
                })
            
            # Start new config
            test_num = int(line.split('Test')[1].split(':')[0].strip())
            config_name = line.split(':')[1].strip()
            
            # Determine config type
            is_cpu = False
            clean_name = "Unknown"
            
            if 'CPU-Only' in config_name or 'CPU-Only' in line:
                clean_name = "CPU-Only"
                is_cpu = True
            elif 'Partial' in config_name:
                clean_name = "GPU Partial"
            elif 'Full' in config_name:
                clean_name = "GPU Full"
            elif 'Single GPU' in config_name:
                clean_name = "Single GPU"
            elif 'Dual GPU' in config_name:
                clean_name = "Dual GPU"
            elif 'Quad GPU' in config_name:
                if 'Balanced' in config_name:
                    clean_name = "Quad GPU (Balanced)"
                elif 'Custom' in config_name:
                    clean_name = "Quad GPU (Custom)"
                else:
                    clean_name = "Quad GPU"
            
            current_config = {
                'name': clean_name,
                'is_cpu': is_cpu,
                'pp512': [],
                'tg128': [],
                'test_num': test_num
            }
            in_results_table = False
        
        # Check for CUDA init failed (CPU-only indicator)
        elif current_config and 'failed to initialize CUDA' in line:
            current_config['is_cpu'] = True
            current_config['name'] = "CPU-Only"
        
        # Detect results table
        elif '| model' in line and 'test' in line and 't/s' in line:
            in_results_table = True
        elif in_results_table and '| qwen3 8B' in line:
            # Parse result line
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 7:
                try:
                    # Find test type (pp512 or tg128)
                    test_type = None
                    tokens_per_sec = None
                    
                    for j, part in enumerate(parts):
                        if 'pp512' in part or 'pp128' in part:
                            test_type = 'pp512'
                            tokens_per_sec = float(parts[j+1].split('±')[0].strip())
                        elif 'tg128' in part or 'tg512' in part:
                            test_type = 'tg128'
                            tokens_per_sec = float(parts[j+1].split('±')[0].strip())
                    
                    if current_config and test_type and tokens_per_sec:
                        current_config[test_type].append(tokens_per_sec)
                except (ValueError, IndexError):
                    pass
    
    # Save last config
    if current_config and (current_config['pp512'] or current_config['tg128']):
        avg_pp = sum(current_config['pp512']) / len(current_config['pp512']) if current_config['pp512'] else 0
        avg_tg = sum(current_config['tg128']) / len(current_config['tg128']) if current_config['tg128'] else 0
        results['configurations'].append({
            'name': current_config['name'],
            'is_cpu_only': current_config['is_cpu'],
            'pp512': avg_pp,
            'tg128': avg_tg,
            'test_num': current_config['test_num']
        })
    
    return results

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
    benchmark_files = sorted(BENCHMARK_DIR.glob("benchmark_results*.md"))
    
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
            if results['configurations']:
                results_list.append(results)
                print(f"    → Parsed {len(results['configurations'])} configurations")
        except Exception as e:
            print(f"    ⚠ Error parsing {filepath.name}: {e}")
    
    if not results_list:
        print("\n❌ No valid results found")
        tee.close()
        sys.stdout = tee.terminal
        return
    
    # === ANALYSIS SECTION ===
    
    print("\n" + "="*70)
    print("DETAILED RESULTS BY CONFIGURATION")
    print("="*70)
    
    # Find CPU baseline
    cpu_baseline = None
    cpu_node = None
    
    for result in results_list:
        for config in result['configurations']:
            if config['is_cpu_only']:
                cpu_baseline = config
                cpu_node = result['node']
                print(f"\n🖥️  CPU-Only Baseline ({result['node']}):")
                print(f"   Prompt Processing: {config['pp512']:.2f} t/s")
                print(f"   Text Generation:   {config['tg128']:.2f} t/s")
                break
        if cpu_baseline:
            break
    
    if not cpu_baseline:
        print("\n⚠️  No CPU baseline found!")
    
    # Print GPU configurations
    print("\n" + "-"*70)
    print("GPU CONFIGURATIONS")
    print("-"*70)
    
    for result in results_list:
        has_gpu_config = any(not c['is_cpu_only'] for c in result['configurations'])
        if has_gpu_config:
            print(f"\n📍 Node: {result['node']}")
            print(f"   GPU: {result['gpu_type'] or 'Unknown'}")
            print(f"   GPU Count: {result['gpu_count'] or 'N/A'}")
            print()
            
            for config in sorted(result['configurations'], key=lambda x: x.get('test_num', 0)):
                if not config['is_cpu_only']:
                    print(f"   {config['name']:25s} | pp512: {config['pp512']:8.2f} t/s | tg128: {config['tg128']:6.2f} t/s")
    
    # === COMPARISON TABLE ===
    
    print("\n" + "="*70)
    print("COMPREHENSIVE COMPARISON TABLE")
    print("="*70)
    print()
    print("| Node      | GPU Type     | Config              | Prompt (pp512) | Generation (tg128) |")
    print("|-----------|--------------|---------------------|----------------|-------------------|")
    
    for result in results_list:
        for config in sorted(result['configurations'], key=lambda x: x.get('test_num', 0)):
            node = result['node'] or 'Unknown'
            gpu_type = result['gpu_type'] or 'CPU'
            config_name = config['name']
            pp512 = config['pp512']
            tg128 = config['tg128']
            
            print(f"| {node:9s} | {gpu_type:12s} | {config_name:19s} | {pp512:14.2f} | {tg128:17.2f} |")
    
    # === SPEEDUP ANALYSIS ===
    
    if cpu_baseline:
        print("\n" + "="*70)
        print("SPEEDUP ANALYSIS")
        print("="*70)
        print(f"\nCPU Baseline ({cpu_node}):")
        print(f"  Prompt Processing: {cpu_baseline['pp512']:.2f} t/s")
        print(f"  Text Generation:   {cpu_baseline['tg128']:.2f} t/s")
        print("\nGPU Speedups:")
        print()
        print("| Node      | GPU Type     | Config              | Prompt Speedup | Generation Speedup |")
        print("|-----------|--------------|---------------------|----------------|-------------------|")
        
        for result in results_list:
            for config in sorted(result['configurations'], key=lambda x: x.get('test_num', 0)):
                if not config['is_cpu_only']:
                    pp_speedup = config['pp512'] / cpu_baseline['pp512']
                    tg_speedup = config['tg128'] / cpu_baseline['tg128']
                    
                    node = result['node'] or 'Unknown'
                    gpu_type = result['gpu_type'] or 'Unknown'
                    config_name = config['name']
                    
                    print(f"| {node:9s} | {gpu_type:12s} | {config_name:19s} | {pp_speedup:14.2f}x | {tg_speedup:17.2f}x |")
    
    # === KEY FINDINGS ===
    
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    
    # Find best GPU configurations
    gpu_configs = []
    for result in results_list:
        for config in result['configurations']:
            if not config['is_cpu_only']:
                gpu_configs.append((result, config))
    
    if gpu_configs:
        best_pp = max(gpu_configs, key=lambda x: x[1]['pp512'])
        best_tg = max(gpu_configs, key=lambda x: x[1]['tg128'])
        
        print(f"\n1. Best Prompt Processing Performance:")
        print(f"   {best_pp[0]['node']} - {best_pp[0]['gpu_type']} - {best_pp[1]['name']}")
        print(f"   {best_pp[1]['pp512']:.2f} t/s")
        if cpu_baseline:
            speedup = best_pp[1]['pp512'] / cpu_baseline['pp512']
            print(f"   Speedup: {speedup:.2f}x vs CPU")
        
        print(f"\n2. Best Text Generation Performance:")
        print(f"   {best_tg[0]['node']} - {best_tg[0]['gpu_type']} - {best_tg[1]['name']}")
        print(f"   {best_tg[1]['tg128']:.2f} t/s")
        if cpu_baseline:
            speedup = best_tg[1]['tg128'] / cpu_baseline['tg128']
            print(f"   Speedup: {speedup:.2f}x vs CPU")
        
        # Multi-GPU analysis
        single_gpu = [x for x in gpu_configs if 'Single GPU' in x[1]['name'] or 'GPU Full' in x[1]['name']]
        multi_gpu = [x for x in gpu_configs if 'Dual GPU' in x[1]['name'] or 'Quad GPU' in x[1]['name']]
        
        if single_gpu and multi_gpu:
            print(f"\n3. Multi-GPU Scaling Analysis:")
            single_avg_pp = sum(x[1]['pp512'] for x in single_gpu) / len(single_gpu)
            multi_avg_pp = sum(x[1]['pp512'] for x in multi_gpu) / len(multi_gpu)
            
            print(f"   Single GPU avg: {single_avg_pp:.2f} t/s (prompt)")
            print(f"   Multi GPU avg:  {multi_avg_pp:.2f} t/s (prompt)")
            
            if multi_avg_pp < single_avg_pp:
                loss = ((single_avg_pp - multi_avg_pp) / single_avg_pp * 100)
                print(f"   ⚠️  Multi-GPU shows NEGATIVE scaling: {loss:.1f}% performance loss")
                print(f"   💡 Recommendation: Use single GPU for this model size")
            else:
                gain = ((multi_avg_pp - single_avg_pp) / single_avg_pp * 100)
                print(f"   ✅ Multi-GPU shows positive scaling: {gain:.1f}% performance gain")
        
        # Hardware comparison
        gpu_types = {}
        for result, config in gpu_configs:
            gpu_type = result['gpu_type']
            if gpu_type:
                if gpu_type not in gpu_types:
                    gpu_types[gpu_type] = []
                gpu_types[gpu_type].append(config['pp512'])
        
        if len(gpu_types) > 1:
            print(f"\n4. Hardware Comparison:")
            for gpu_type, values in sorted(gpu_types.items()):
                avg = sum(values) / len(values)
                print(f"   {gpu_type}: {avg:.2f} t/s (avg prompt processing)")
    
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
