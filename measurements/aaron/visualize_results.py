#!/usr/bin/env python3
"""
Visualization script - reads analysis_*.md files
"""

import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path

BENCHMARK_DIR = Path.home() / "perf-analysis-modeling-project/measurements/aaron"
OUTPUT_DIR = BENCHMARK_DIR / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

plt.style.use('seaborn-v0_8-darkgrid')
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#8B5A3C']

def find_latest_analysis():
    files = sorted(BENCHMARK_DIR.glob("analysis_*.md"))
    if not files:
        raise FileNotFoundError("No analysis files found")
    return files[-1]

def parse_analysis(filepath):
    with open(filepath) as f:
        content = f.read()
    
    data = {'cpu': None, 'configs': []}
    
    # Extract CPU baseline
    cpu_match = re.search(r'CPU Baseline.*?Prompt Processing: ([\d.]+) t/s.*?Text Generation:\s+([\d.]+) t/s', content, re.DOTALL)
    if cpu_match:
        data['cpu'] = {
            'pp512': float(cpu_match.group(1)),
            'tg128': float(cpu_match.group(2))
        }
    
    # Extract from comparison table
    table_section = re.search(r'COMPREHENSIVE COMPARISON TABLE.*?\n\n(.*?)\n\n', content, re.DOTALL)
    if table_section:
        lines = table_section.group(1).split('\n')
        for line in lines:
            if '|' in line and 'Node' not in line and '---' not in line:
                parts = [p.strip() for p in line.split('|')[1:-1]]
                if len(parts) >= 5:
                    try:
                        data['configs'].append({
                            'node': parts[0],
                            'gpu_type': parts[1],
                            'name': parts[2],
                            'pp512': float(parts[3]),
                            'tg128': float(parts[4]),
                            'is_cpu': 'CPU-Only' in parts[2]
                        })
                    except (ValueError, IndexError):
                        pass
    
    return data

def create_visualizations(data):
    if not data['cpu']:
        print("⚠ No CPU baseline")
        return
    
    cpu = data['cpu']
    configs = data['configs']
    
    cpu_configs = [c for c in configs if c['is_cpu']]
    gpu_configs = [c for c in configs if not c['is_cpu']]
    
    # === FIGURE 1: CPU vs GPU ===
    if cpu_configs and gpu_configs:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('CPU vs GPU Performance', fontsize=16, fontweight='bold')
        
        plot_configs = cpu_configs[:1]
        for c in gpu_configs:
            if 'Partial' in c['name'] or 'Full' in c['name']:
                plot_configs.append(c)
                if len(plot_configs) >= 3:
                    break
        
        names = [c['name'].replace(' ', '\n') for c in plot_configs]
        pp_vals = [c['pp512'] for c in plot_configs]
        tg_vals = [c['tg128'] for c in plot_configs]
        
        bars = ax1.bar(names, pp_vals, color=colors[:len(names)], edgecolor='black', linewidth=1.2)
        ax1.set_ylabel('Tokens/Second', fontsize=12, fontweight='bold')
        ax1.set_title('Prompt Processing (pp512)', fontsize=13, fontweight='bold')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., h, f'{h:.1f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        bars = ax2.bar(names, tg_vals, color=colors[:len(names)], edgecolor='black', linewidth=1.2)
        ax2.set_ylabel('Tokens/Second', fontsize=12, fontweight='bold')
        ax2.set_title('Text Generation (tg128)', fontsize=13, fontweight='bold')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., h, f'{h:.1f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/1_cpu_vs_gpu.png', dpi=300, bbox_inches='tight')
        print('✓ Saved: 1_cpu_vs_gpu.png')
        plt.close()
    
    # === FIGURE 2: Speedup ===
    if gpu_configs:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        plot_configs = gpu_configs[:4]
        names = [c['name'].replace(' ', '\n') for c in plot_configs]
        pp_speedup = [c['pp512'] / cpu['pp512'] for c in plot_configs]
        tg_speedup = [c['tg128'] / cpu['tg128'] for c in plot_configs]
        
        x = np.arange(len(names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, pp_speedup, width, label='Prompt',
                      color=colors[0], edgecolor='black', linewidth=1.2)
        bars2 = ax.bar(x + width/2, tg_speedup, width, label='Generation',
                      color=colors[1], edgecolor='black', linewidth=1.2)
        
        ax.set_ylabel('Speedup vs CPU', fontsize=12, fontweight='bold')
        ax.set_title('GPU Acceleration Speedup', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.legend(fontsize=11)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y')
        
        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., h, f'{h:.0f}x',
                       ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/2_speedup.png', dpi=300, bbox_inches='tight')
        print('✓ Saved: 2_speedup.png')
        plt.close()
    
    # === FIGURE 3: Multi-GPU ===
    multi_gpu = [c for c in gpu_configs if 'Dual' in c['name'] or 'Quad' in c['name'] or 'Single GPU' in c['name']]
    if len(multi_gpu) > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Multi-GPU Scaling', fontsize=16, fontweight='bold')
        
        names = [c['name'].replace(' ', '\n') for c in multi_gpu]
        pp_vals = [c['pp512'] for c in multi_gpu]
        tg_vals = [c['tg128'] for c in multi_gpu]
        
        bars = ax1.bar(names, pp_vals, color=colors[:len(names)], edgecolor='black', linewidth=1.2)
        ax1.set_ylabel('Tokens/Second', fontsize=12, fontweight='bold')
        ax1.set_title('Prompt Processing', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        if 'Single GPU' in multi_gpu[0]['name']:
            ax1.axhline(y=pp_vals[0], color='red', linestyle='--', linewidth=2, label='Single GPU')
            ax1.legend()
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., h, f'{h:.0f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        bars = ax2.bar(names, tg_vals, color=colors[:len(names)], edgecolor='black', linewidth=1.2)
        ax2.set_ylabel('Tokens/Second', fontsize=12, fontweight='bold')
        ax2.set_title('Text Generation', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        if 'Single GPU' in multi_gpu[0]['name']:
            ax2.axhline(y=tg_vals[0], color='red', linestyle='--', linewidth=2, label='Single GPU')
            ax2.legend()
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., h, f'{h:.1f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/3_multi_gpu.png', dpi=300, bbox_inches='tight')
        print('✓ Saved: 3_multi_gpu.png')
        plt.close()
    
    # === FIGURE 4: Hardware Comparison ===
    gpu_types = {}
    for c in gpu_configs:
        if 'Full' in c['name'] or 'Single GPU' in c['name']:
            gpu = c['gpu_type']
            if gpu not in gpu_types or c['pp512'] > gpu_types[gpu]['pp512']:
                gpu_types[gpu] = c
    
    if len(gpu_types) >= 2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Hardware Comparison', fontsize=16, fontweight='bold')
        
        names = [gpu.replace('NVIDIA ', '') for gpu in gpu_types.keys()]
        pp_vals = [c['pp512'] for c in gpu_types.values()]
        tg_vals = [c['tg128'] for c in gpu_types.values()]
        
        bars = ax1.bar(names, pp_vals, color=colors[:len(names)], edgecolor='black', linewidth=1.2)
        ax1.set_ylabel('Tokens/Second', fontsize=12, fontweight='bold')
        ax1.set_title('Prompt Processing', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., h, f'{h:.0f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        speedup = max(pp_vals) / min(pp_vals)
        ax1.text(0.5, max(pp_vals)*0.5, f'{speedup:.2f}x faster',
                ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        bars = ax2.bar(names, tg_vals, color=colors[:len(names)], edgecolor='black', linewidth=1.2)
        ax2.set_ylabel('Tokens/Second', fontsize=12, fontweight='bold')
        ax2.set_title('Text Generation', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., h, f'{h:.1f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/4_hardware.png', dpi=300, bbox_inches='tight')
        print('✓ Saved: 4_hardware.png')
        plt.close()

def main():
    print("="*70)
    print("VISUALIZATION GENERATION")
    print("="*70)
    
    try:
        analysis_file = find_latest_analysis()
        print(f"\n✓ Found: {analysis_file.name}")
    except FileNotFoundError as e:
        print(f"\n✗ {e}")
        return
    
    print("\nParsing analysis file...")
    data = parse_analysis(analysis_file)
    
    if data['cpu']:
        print(f"✓ CPU baseline: {data['cpu']['pp512']:.2f} t/s")
    print(f"✓ Found {len(data['configs'])} configurations")
    
    print(f"\nGenerating figures in: {OUTPUT_DIR}")
    print("-" * 70)
    
    create_visualizations(data)
    
    print("-" * 70)
    print("\n✅ COMPLETE!")
    print(f"\nFigures saved to: {OUTPUT_DIR}")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  • {f.name}")
    print("="*70)

if __name__ == "__main__":
    main()
