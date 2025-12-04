#!/usr/bin/env python3
"""
Dynamic Benchmark Analysis Script - FIXED VERSION
Parses test sections separately instead of averaging all results
"""

import re
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
    """Parse benchmark file by extracting each test section separately"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    results = {
        'filepath': str(filepath),
        'filename': filepath.name,
        'node': None,
        'gpu_type': None,
        'gpu_count': None,
        'configurations': []  # List of test configurations
    }
    
    # Extract node information
    node_match = re.search(r'\*\*Node:\*\* (\w+)', content)
    if node_match:
        results['node'] = node_match.group(1)
    
    # Extract GPU count
    gpu_count_match = re.search(r'\*\*GPUs per Node:\*\* (\d+)', content)
    if gpu_count_match:
        results['gpu_count'] = int(gpu_count_match.group(1))
    
    # Extract GPU type
    gpu_match = re.search(r'Device \d+: (NVIDIA \w+)', content)
    if gpu_match:
        results['gpu_type'] = gpu_match.group(1)
    
    # Split content into test sections - IMPROVED REGEX
    # Look for "## Test N:" markers with the configuration name
    test_pattern = r'## Test (\d+):\s*(.+?)\n\n\*\*Configuration:\*\*(.+?)```'
    test_matches = re.finditer(test_pattern, content, re.DOTALL)
    
    for match in test_matches:
        test_num = int(match.group(1))
        section_title = match.group(2).strip()
        section_content = match.group(3)
        
        # Determine configuration name from section title
        config_name = "Unknown"
        is_cpu_only = False
        
        if 'CPU-Only' in section_title:
            config_name = "CPU-Only"
            is_cpu_only = True
        elif 'Partial' in section_title:
            config_name = "GPU Partial"
        elif 'Full' in section_title:
            config_name = "GPU Full"
        elif 'Single GPU' in section_title:
            config_name = "Single GPU"
        elif 'Dual GPU' in section_title:
            config_name = "Dual GPU"
        elif 'Quad GPU' in section_title and 'Balanced' in section_title:
            config_name = "Quad GPU (Balanced)"
        elif 'Quad GPU' in section_title and 'Custom' in section_title:
            config_name = "Quad GPU (Custom)"
        
        # Now get the full section content to extract performance data
        # Find the section starting from this test header
        section_start = match.start()
        # Find the next test header or end of file
        next_test = re.search(r'## Test \d+:', content[match.end():])
        if next_test:
            section_end = match.end() + next_test.start()
        else:
            # Check for "## Performance Summary" or similar
            summary_match = re.search(r'## Performance Summary|## Observations|## Build Configuration', content[match.end():])
            if summary_match:
                section_end = match.end() + summary_match.start()
            else:
                section_end = len(content)
        
        section = content[section_start:section_end]
        
        # Check if CPU-only from error message
        if 'failed to initialize CUDA' in section:
            is_cpu_only = True
            config_name = "CPU-Only"
        
        # Extract performance numbers from this section
        test_results = {'pp512': [], 'tg128': []}
        pattern = r'\|\s*qwen3 8B.*?\|\s*[\d.]+\s*GiB\s*\|\s*[\d.]+\s*B\s*\|\s*[\w,]+\s*\|\s*\d+\s*\|(?:\s*[\d.]+\s*\|)?\s*(pp\d+|tg\d+)\s*\|\s*([\d.]+)'
        
        matches = re.finditer(pattern, section)
        for m in matches:
            test_type = m.group(1)
            tokens_per_sec = float(m.group(2))
            test_results[test_type].append(tokens_per_sec)
        
        # Calculate averages for this configuration
        if test_results['pp512'] or test_results['tg128']:
            avg_pp512 = sum(test_results['pp512']) / len(test_results['pp512']) if test_results['pp512'] else 0
            avg_tg128 = sum(test_results['tg128']) / len(test_results['tg128']) if test_results['tg128'] else 0
            
            results['configurations'].append({
                'name': config_name,
                'is_cpu_only': is_cpu_only,
                'pp512': avg_pp512,
                'tg128': avg_tg128,
                'test_num': test_num
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
        except Exception as e:
            print(f"    ⚠ Error parsing {filepath.name}: {e}")
            import traceback
            traceback.print_exc()
    
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
