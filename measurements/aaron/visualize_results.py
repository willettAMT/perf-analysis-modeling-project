#!/usr/bin/env python3
"""
Performance Visualization Script for Qwen3-8B Benchmarks
CS 431/531 Performance Project
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Set output directory
OUTPUT_DIR = os.path.expanduser('~/perf-analysis-modeling-project/measurements/aaron/figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set style for professional-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']

# ============================================================================
# DATA FROM BENCHMARKS
# ============================================================================

# 1-GPU Configuration Data (L40S - orcaga02)
single_gpu_configs = ['CPU-Only\n(64 threads)', 'GPU Partial\n(10 layers)', 'GPU Full\n(all layers)']
single_gpu_prompt = [7.12, 9.76, 7698.65]
single_gpu_generation = [15.84, 19.67, 104.61]

# 4-GPU Configuration Data (L40S - orcaga01)
multi_gpu_configs = ['1 GPU', '2 GPUs', '4 GPUs\n(Balanced)', '4 GPUs\n(Custom)']
multi_gpu_prompt = [7907.81, 7820.44, 7758.68, 7684.82]
multi_gpu_generation = [104.18, 103.80, 104.46, 104.48]

# Hardware Comparison (A30 vs L40S)
hardware_names = ['A30\n(orcaga10)', 'L40S\n(orcaga02)']
hardware_prompt = [2423.23, 7907.81]
hardware_generation = [75.60, 104.18]

# ============================================================================
# FIGURE 1: Single GPU Configuration Comparison
# ============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Single GPU: CPU vs GPU Offloading Performance', fontsize=16, fontweight='bold')

# Prompt Processing
bars1 = ax1.bar(single_gpu_configs, single_gpu_prompt, color=colors[:3], edgecolor='black', linewidth=1.2)
ax1.set_ylabel('Tokens/Second', fontsize=12, fontweight='bold')
ax1.set_title('Prompt Processing (pp512)', fontsize=13, fontweight='bold')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}',
             ha='center', va='bottom', fontweight='bold', fontsize=10)

# Text Generation
bars2 = ax2.bar(single_gpu_configs, single_gpu_generation, color=colors[:3], edgecolor='black', linewidth=1.2)
ax2.set_ylabel('Tokens/Second', fontsize=12, fontweight='bold')
ax2.set_title('Text Generation (tg128)', fontsize=13, fontweight='bold')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3)

# Add value labels on bars
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}',
             ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/1_single_gpu_comparison.png', dpi=300, bbox_inches='tight')
print(f'✓ Saved: {OUTPUT_DIR}/1_single_gpu_comparison.png')
plt.close()

# ============================================================================
# FIGURE 2: Speedup Analysis
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

cpu_baseline_prompt = 7.12
cpu_baseline_gen = 15.84

speedup_configs = ['GPU Partial\n(10 layers)', 'GPU Full\n(all layers)']
speedup_prompt = [9.76/cpu_baseline_prompt, 7698.65/cpu_baseline_prompt]
speedup_gen = [19.67/cpu_baseline_gen, 104.61/cpu_baseline_gen]

x = np.arange(len(speedup_configs))
width = 0.35

bars1 = ax.bar(x - width/2, speedup_prompt, width, label='Prompt Processing', 
               color=colors[0], edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x + width/2, speedup_gen, width, label='Text Generation',
               color=colors[1], edgecolor='black', linewidth=1.2)

ax.set_ylabel('Speedup (vs CPU-Only)', fontsize=12, fontweight='bold')
ax.set_title('GPU Acceleration Speedup over CPU-Only Baseline', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(speedup_configs)
ax.legend(fontsize=11)
ax.set_yscale('log')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}x',
                ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/2_speedup_analysis.png', dpi=300, bbox_inches='tight')
print(f'✓ Saved: {OUTPUT_DIR}/2_speedup_analysis.png')
plt.close()

# ============================================================================
# FIGURE 3: Multi-GPU Scaling Analysis
# ============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Multi-GPU Scaling: Does More GPUs = Better Performance?', fontsize=16, fontweight='bold')

# Prompt Processing
bars1 = ax1.bar(multi_gpu_configs, multi_gpu_prompt, color=colors[:4], edgecolor='black', linewidth=1.2)
ax1.set_ylabel('Tokens/Second', fontsize=12, fontweight='bold')
ax1.set_title('Prompt Processing (pp512)', fontsize=13, fontweight='bold')
ax1.set_ylim([7600, 8000])
ax1.grid(True, alpha=0.3)
ax1.axhline(y=multi_gpu_prompt[0], color='red', linestyle='--', linewidth=2, label='1-GPU Baseline')
ax1.legend()

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.0f}',
             ha='center', va='bottom', fontweight='bold', fontsize=9)

# Text Generation
bars2 = ax2.bar(multi_gpu_configs, multi_gpu_generation, color=colors[:4], edgecolor='black', linewidth=1.2)
ax2.set_ylabel('Tokens/Second', fontsize=12, fontweight='bold')
ax2.set_title('Text Generation (tg128)', fontsize=13, fontweight='bold')
ax2.set_ylim([103, 105])
ax2.grid(True, alpha=0.3)
ax2.axhline(y=multi_gpu_generation[0], color='red', linestyle='--', linewidth=2, label='1-GPU Baseline')
ax2.legend()

# Add value labels
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}',
             ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/3_multi_gpu_scaling.png', dpi=300, bbox_inches='tight')
print(f'✓ Saved: {OUTPUT_DIR}/3_multi_gpu_scaling.png')
plt.close()

# ============================================================================
# FIGURE 4: Multi-GPU Efficiency
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

gpu_counts = [1, 2, 4, 4]
efficiency_prompt = [100, 98.9, 98.1, 97.2]  # Percentage of 1-GPU performance
efficiency_gen = [100, 99.6, 100.3, 100.3]

x = np.arange(len(multi_gpu_configs))
width = 0.35

bars1 = ax.bar(x - width/2, efficiency_prompt, width, label='Prompt Processing',
               color=colors[0], edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x + width/2, efficiency_gen, width, label='Text Generation',
               color=colors[1], edgecolor='black', linewidth=1.2)

ax.set_ylabel('Performance (% of 1-GPU Baseline)', fontsize=12, fontweight='bold')
ax.set_title('Multi-GPU Efficiency: Performance vs Single GPU', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(multi_gpu_configs)
ax.legend(fontsize=11)
ax.set_ylim([95, 102])
ax.axhline(y=100, color='red', linestyle='--', linewidth=2, label='100% (1-GPU baseline)')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/4_multi_gpu_efficiency.png', dpi=300, bbox_inches='tight')
print(f'✓ Saved: {OUTPUT_DIR}/4_multi_gpu_efficiency.png')
plt.close()

# ============================================================================
# FIGURE 5: Hardware Comparison (A30 vs L40S)
# ============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Hardware Comparison: A30 vs L40S GPU Performance', fontsize=16, fontweight='bold')

# Prompt Processing
bars1 = ax1.bar(hardware_names, hardware_prompt, color=[colors[2], colors[0]], edgecolor='black', linewidth=1.2)
ax1.set_ylabel('Tokens/Second', fontsize=12, fontweight='bold')
ax1.set_title('Prompt Processing (pp512)', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Add value labels and speedup
for i, bar in enumerate(bars1):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.0f} t/s',
             ha='center', va='bottom', fontweight='bold', fontsize=11)

speedup = hardware_prompt[1] / hardware_prompt[0]
ax1.text(0.5, max(hardware_prompt) * 0.5, f'L40S: {speedup:.2f}x faster',
         ha='center', fontsize=13, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# Text Generation
bars2 = ax2.bar(hardware_names, hardware_generation, color=[colors[2], colors[0]], edgecolor='black', linewidth=1.2)
ax2.set_ylabel('Tokens/Second', fontsize=12, fontweight='bold')
ax2.set_title('Text Generation (tg128)', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Add value labels and speedup
for i, bar in enumerate(bars2):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f} t/s',
             ha='center', va='bottom', fontweight='bold', fontsize=11)

speedup_gen = hardware_generation[1] / hardware_generation[0]
ax2.text(0.5, max(hardware_generation) * 0.5, f'L40S: {speedup_gen:.2f}x faster',
         ha='center', fontsize=13, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/5_hardware_comparison.png', dpi=300, bbox_inches='tight')
print(f'✓ Saved: {OUTPUT_DIR}/5_hardware_comparison.png')
plt.close()

# ============================================================================
# FIGURE 6: Summary Dashboard
# ============================================================================

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Title
fig.suptitle('Qwen3-8B Performance Analysis Dashboard', fontsize=18, fontweight='bold', y=0.98)

# Subplot 1: CPU vs GPU Comparison
ax1 = fig.add_subplot(gs[0, 0])
configs_simple = ['CPU', 'GPU']
prompt_simple = [7.12, 7698.65]
bars = ax1.bar(configs_simple, prompt_simple, color=[colors[2], colors[0]], edgecolor='black', linewidth=1.2)
ax1.set_ylabel('Tokens/Second', fontweight='bold')
ax1.set_title('Prompt Processing: CPU vs GPU', fontweight='bold', fontsize=12)
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}',
             ha='center', va='bottom', fontweight='bold')

# Subplot 2: Speedup Summary
ax2 = fig.add_subplot(gs[0, 1])
speedup_data = [1081, 6.6]
speedup_labels = ['Prompt\nProcessing', 'Text\nGeneration']
bars = ax2.bar(speedup_labels, speedup_data, color=[colors[0], colors[1]], edgecolor='black', linewidth=1.2)
ax2.set_ylabel('Speedup (vs CPU)', fontweight='bold')
ax2.set_title('GPU Acceleration Speedup', fontweight='bold', fontsize=12)
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height, f'{height:.0f}x',
             ha='center', va='bottom', fontweight='bold')

# Subplot 3: Multi-GPU Scaling
ax3 = fig.add_subplot(gs[1, :])
gpu_labels = ['1 GPU', '2 GPUs', '4 GPUs\n(Balanced)', '4 GPUs\n(Custom)']
bars = ax3.bar(gpu_labels, multi_gpu_prompt, color=colors[:4], edgecolor='black', linewidth=1.2)
ax3.set_ylabel('Tokens/Second', fontweight='bold')
ax3.set_title('Multi-GPU Scaling: Prompt Processing Performance', fontweight='bold', fontsize=12)
ax3.set_ylim([7600, 8000])
ax3.axhline(y=multi_gpu_prompt[0], color='red', linestyle='--', linewidth=2, alpha=0.7)
ax3.grid(True, alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height, f'{height:.0f}',
             ha='center', va='bottom', fontweight='bold')

# Subplot 4: Hardware Comparison
ax4 = fig.add_subplot(gs[2, 0])
bars = ax4.bar(hardware_names, hardware_prompt, color=[colors[2], colors[0]], edgecolor='black', linewidth=1.2)
ax4.set_ylabel('Tokens/Second', fontweight='bold')
ax4.set_title('Hardware: A30 vs L40S (Prompt)', fontweight='bold', fontsize=12)
ax4.grid(True, alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height, f'{height:.0f}',
             ha='center', va='bottom', fontweight='bold')

# Subplot 5: Key Findings Text
ax5 = fig.add_subplot(gs[2, 1])
ax5.axis('off')
findings_text = """
KEY FINDINGS

✓ GPU acceleration provides 1081x speedup
   for prompt processing

✓ Multi-GPU provides NO benefit for 8B model
   (slight degradation due to overhead)

✓ L40S outperforms A30 by 3.26x

✓ Single GPU with full offload is optimal

✓ Partial offloading is worst option
   (CPU↔GPU transfer overhead)

RECOMMENDATION:
Use 1 GPU with full layer offload
for maximum inference performance
"""
ax5.text(0.1, 0.9, findings_text, transform=ax5.transAxes,
         fontsize=11, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.savefig(f'{OUTPUT_DIR}/6_summary_dashboard.png', dpi=300, bbox_inches='tight')
print(f'✓ Saved: {OUTPUT_DIR}/6_summary_dashboard.png')
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*60)
print("VISUALIZATION GENERATION COMPLETE!")
print("="*60)
print(f"\nAll figures saved to: {OUTPUT_DIR}")
print("\nGenerated visualizations:")
print("  1. 1_single_gpu_comparison.png - CPU vs GPU offloading")
print("  2. 2_speedup_analysis.png - Speedup metrics")
print("  3. 3_multi_gpu_scaling.png - Multi-GPU performance")
print("  4. 4_multi_gpu_efficiency.png - Multi-GPU efficiency")
print("  5. 5_hardware_comparison.png - A30 vs L40S comparison")
print("  6. 6_summary_dashboard.png - Complete overview dashboard")
print("\nUse these figures in your project presentation/report!")
print("="*60)
