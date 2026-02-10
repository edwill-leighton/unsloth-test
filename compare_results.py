#!/usr/bin/env python3
"""
Compare benchmark results across GPUs.

Usage:
    python compare_results.py work/benchmark_*.json
    python compare_results.py results_5090.json results_dgx.json results_pro6000.json
"""

import json
import sys
import os
import glob


def load_results(paths: list[str]) -> list[dict]:
    results = []
    for pattern in paths:
        for path in glob.glob(pattern):
            with open(path) as f:
                results.append(json.load(f))
    return results


def print_comparison(results: list[dict]):
    if not results:
        print("No results found. Pass JSON result files as arguments.")
        print("Example: python compare_results.py work/benchmark_*.json")
        return

    # Group by profile
    profiles = {}
    for r in results:
        key = r.get("profile", "unknown")
        if key not in profiles:
            profiles[key] = []
        profiles[key].append(r)

    for profile_name, runs in sorted(profiles.items()):
        print("\n" + "=" * 90)
        print(f"  PROFILE: {profile_name.upper()}  —  {runs[0]['config']['model']}")
        print("=" * 90)

        # Header
        header = f"{'Metric':<30}"
        for r in runs:
            gpu = r["gpu"]["gpu_name"][:20]
            header += f" | {gpu:>20}"
        print(header)
        print("-" * len(header))

        # VRAM
        row = f"{'VRAM Total (GB)':<30}"
        for r in runs:
            row += f" | {r['gpu']['vram_total_gb']:>20}"
        print(row)

        row = f"{'VRAM Peak (GB)':<30}"
        for r in runs:
            row += f" | {r['vram']['peak_training_gb']:>20}"
        print(row)

        row = f"{'VRAM Utilization (%)':<30}"
        for r in runs:
            row += f" | {r['vram']['utilization_pct']:>19}%"
        print(row)

        print("-" * len(header))

        # Timing
        row = f"{'Model Load Time (s)':<30}"
        for r in runs:
            row += f" | {r['timing']['model_load_sec']:>20}"
        print(row)

        row = f"{'Training Time (s)':<30}"
        for r in runs:
            row += f" | {r['timing']['training_sec']:>20}"
        print(row)

        row = f"{'Total Time (s)':<30}"
        for r in runs:
            row += f" | {r['timing']['total_sec']:>20}"
        print(row)

        print("-" * len(header))

        # Throughput
        row = f"{'Samples/sec':<30}"
        for r in runs:
            row += f" | {r['throughput']['samples_per_second']:>20}"
        print(row)

        row = f"{'Tokens/sec':<30}"
        for r in runs:
            row += f" | {r['throughput']['tokens_per_second']:>20}"
        print(row)

        row = f"{'Steps/sec':<30}"
        for r in runs:
            row += f" | {r['throughput']['steps_per_second']:>20}"
        print(row)

        print("-" * len(header))

        # Loss
        row = f"{'Final Loss':<30}"
        for r in runs:
            loss = r["training"]["final_loss"]
            if isinstance(loss, float):
                row += f" | {loss:>20.4f}"
            else:
                row += f" | {str(loss):>20}"
        print(row)

        # Speedup relative to first GPU
        if len(runs) > 1:
            print("-" * len(header))
            base_time = runs[0]["timing"]["training_sec"]
            base_tps = runs[0]["throughput"]["tokens_per_second"]
            label = "Speedup vs " + runs[0]['gpu']['gpu_name'][:14]
            row = f"{label:<30}"
            for r in runs:
                speedup = base_time / r["timing"]["training_sec"] if r["timing"]["training_sec"] > 0 else 0
                row += f" | {speedup:>19.2f}x"
            print(row)

        print("=" * 90)


def main():
    if len(sys.argv) < 2:
        # Try to auto-find results
        patterns = ["work/benchmark_*.json", "benchmark_*.json", "*.json"]
        paths = []
        for p in patterns:
            found = glob.glob(p)
            if found:
                paths = found
                break
        if not paths:
            print("Usage: python compare_results.py <result1.json> [result2.json] ...")
            print("       python compare_results.py work/benchmark_*.json")
            return
    else:
        paths = sys.argv[1:]

    results = load_results(paths)
    print(f"\nLoaded {len(results)} benchmark result(s)")
    print_comparison(results)


if __name__ == "__main__":
    main()
