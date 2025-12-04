#!/usr/bin/env python3
"""Simple benchmark parser - matches EXACT file format"""

import re
from pathlib import Path
from datetime import datetime

BENCHMARK_DIR = Path.home() / "perf-analysis-modeling-project/measurements/aaron"
OUTPUT_FILE = BENCHMARK_DIR / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

def extract_number(text):
    """Extract float from text like '7.07 ± 0.08'"""
    match = re.search(r'([\d.]+)', text)
    return float(match.group(1)) if match else 0.0

def parse_file(filepath):
    """Parse one benchmark file"""
    with open(filepath) as f:
        content = f.read()
    
    # Extract metadata
    node = re.search(r'\*\*Node:\*\* (\w+)', content)
    gpu_count = re.search(r'\*\*GPUs per Node:\*\* (\d+)', content)
    gpu_type = re.search(r'Device 0: (NVIDIA \w+)', content)
    
    result = {
        'file': filepath.name,
        'node': node.group(1) if node else 'Unknown',
        'gpu_count': int(gpu_count.group(1)) if gpu_count else 1,
        'gpu_type': gpu_type.group(1) if gpu_type else None,
        'tests': []
    }
    
    # Split by test sections
    test_sections = re.split(r'## Test \d+:', content)[1:]  # Skip header
    
    for i, section in enumerate(test_sections, 1):
        # Get test name from first line
        first_line = section.split('\n')[0].strip()
        
        # Determine config name
        if 'CPU-Only' in first_line or 'failed to initialize CUDA' in section:
            name = 'CPU-Only'
            is_cpu = True
        elif 'Partial' in first_line:
            name = 'GPU Partial'
            is_cpu = False
        elif 'Full' in first_line:
            name = 'GPU Full'
            is_cpu = False
        elif 'Single GPU' in first_line:
            name = 'Single GPU'
            is_cpu = False
        elif 'Dual GPU' in first_line:
            name = 'Dual GPU'
            is_cpu = False
        elif 'Quad GPU' in first_line and 'Balanced' in first_line:
            name = 'Quad GPU (Balanced)'
            is_cpu = False
        elif 'Quad GPU' in first_line and 'Custom' in first_line:
            name = 'Quad GPU (Custom)'
            is_cpu = False
        else:
            name = f'Test {i}'
            is_cpu = False
        
        # Extract all performance numbers
        pp_vals = []
        tg_vals = []
        
        # Find all table rows with performance data
        for line in section.split('\n'):
            if '| qwen3 8B' in line and '|' in line:
                parts = line.split('|')
                # Look for pp512 or tg128 in the line
                if 'pp512' in line:
                    # t/s value is typically the last column
                    for part in reversed(parts):
                        if '±' in part or re.search(r'^\s*[\d.]+', part):
                            val = extract_number(part)
                            if val > 0:
                                pp_vals.append(val)
                                break
                elif 'tg128' in line:
                    for part in reversed(parts):
                        if '±' in part or re.search(r'^\s*[\d.]+', part):
                            val = extract_number(part)
                            if val > 0:
                                tg_vals.append(val)
                                break
        
        if pp_vals or tg_vals:
            result['tests'].append({
                'name': name,
                'is_cpu': is_cpu,
                'pp512': sum(pp_vals) / len(pp_vals) if pp_vals else 0,
                'tg128': sum(tg_vals) / len(tg_vals) if tg_vals else 0
            })
    
    return result

def main():
    print("="*70)
    print("PARSING BENCHMARK FILES")
    print("="*70)
    
    files = sorted(BENCHMARK_DIR.glob("benchmark_results*.md"))
    print(f"\nFound {len(files)} files\n")
    
    all_results = []
    for f in files:
        print(f"Parsing: {f.name}")
        r = parse_file(f)
        print(f"  → {len(r['tests'])} tests found")
        all_results.append(r)
    
    # Find CPU baseline
    cpu = None
    for r in all_results:
        for t in r['tests']:
            if t['is_cpu']:
                cpu = t
                print(f"\n✓ CPU Baseline: {cpu['pp512']:.2f} t/s (pp512)")
                break
        if cpu:
            break
    
    if not cpu:
        print("\n✗ NO CPU BASELINE FOUND!")
        return
    
    # Write analysis
    with open(OUTPUT_FILE, 'w') as out:
        out.write("# Qwen3-8B Benchmark Analysis\n\n")
        out.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        out.write("## CPU Baseline\n\n")
        out.write(f"- Prompt Processing: {cpu['pp512']:.2f} t/s\n")
        out.write(f"- Text Generation: {cpu['tg128']:.2f} t/s\n\n")
        
        out.write("## All Configurations\n\n")
        out.write("| Node | GPU Type | Config | pp512 (t/s) | tg128 (t/s) | Speedup (pp) | Speedup (tg) |\n")
        out.write("|------|----------|--------|-------------|-------------|--------------|---------------|\n")
        
        for r in all_results:
            for t in r['tests']:
                pp_speedup = t['pp512'] / cpu['pp512'] if cpu['pp512'] > 0 else 0
                tg_speedup = t['tg128'] / cpu['tg128'] if cpu['tg128'] > 0 else 0
                gpu = r['gpu_type'] or 'CPU'
                
                out.write(f"| {r['node']} | {gpu} | {t['name']} | ")
                out.write(f"{t['pp512']:.2f} | {t['tg128']:.2f} | ")
                out.write(f"{pp_speedup:.1f}x | {tg_speedup:.1f}x |\n")
    
    print(f"\n✓ Analysis saved: {OUTPUT_FILE.name}")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for r in all_results:
        print(f"\n{r['node']} ({r['gpu_type'] or 'CPU'}):")
        for t in r['tests']:
            pp_speedup = t['pp512'] / cpu['pp512'] if cpu['pp512'] > 0 else 0
            print(f"  {t['name']:25s}: {t['pp512']:8.1f} t/s ({pp_speedup:6.1f}x)")

if __name__ == "__main__":
    main()
