#!/usr/bin/env python3
"""
Capture QRNG Data from --quantum-verbose Output

This script parses the verbose output from llama-cli when using --quantum-verbose
to extract entropy, EDT temperature, QRNG mode, count, and rarity metrics.

Usage:
    # From stdin (pipe llama-cli output)
    ./llama-cli -m model.gguf -p "prompt" -n 100 --quantum-verbose 2>&1 | python capture_qrng_data.py

    # From file
    python capture_qrng_data.py -f quantum_output.txt

    # Demo mode (synthetic data)
    python capture_qrng_data.py --demo

    # Save analysis to JSON
    python capture_qrng_data.py -f quantum_output.txt -o analysis.json
"""

import sys
import re
import json
import argparse
import random
from pathlib import Path
from typing import Optional, TextIO

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from qrng_monitor import QRNGMonitor
from trajectory_analyzer import TrajectoryAnalyzer
from chaos_detector import ChaosDetector


# Regex patterns for parsing --quantum-verbose output
PATTERNS = {
    # Basic entropy line: "llama_sampler_dist_apply: entropy=2.45, threshold=1.50"
    'entropy_basic': re.compile(
        r'llama_sampler_dist_apply:\s+entropy=([0-9.]+),\s+threshold=([0-9.]+)'
    ),
    # Low entropy (greedy): "llama_sampler_dist_apply: LOW ENTROPY (0.85 < 1.50) -> greedy"
    'low_entropy': re.compile(
        r'llama_sampler_dist_apply:\s+LOW ENTROPY\s+\(([0-9.]+)\s*<\s*([0-9.]+)\)\s*->\s*greedy'
    ),
    # High entropy (EDT + QRNG): "llama_sampler_dist_apply: HIGH ENTROPY (2.45 >= 1.50) -> EDT temp=0.85, QRNG"
    'high_entropy': re.compile(
        r'llama_sampler_dist_apply:\s+HIGH ENTROPY\s+\(([0-9.]+)\s*>=\s*([0-9.]+)\)\s*->\s*EDT temp=([0-9.]+),\s*QRNG'
    ),
    # QRNG mode line: "llama_sampler_dist_apply: QRNG mode=1 count=42 (rare)"
    'qrng_mode': re.compile(
        r'llama_sampler_dist_apply:\s+QRNG mode=(\d+)\s+count=(\d+)\s+\((\w+)\)'
    ),
    # Token output (optional): "token: 1234"
    'token': re.compile(r'token:\s*(\d+)'),
}


class QRNGDataCapture:
    """Captures and analyzes QRNG data from quantum-verbose output."""

    def __init__(self):
        self.samples = []
        self.tokens = []
        self.qrng_bytes = []

        # Analysis modules
        self.qrng_monitor = QRNGMonitor(history_len=100)
        self.trajectory_analyzer = TrajectoryAnalyzer()
        self.chaos_detector = ChaosDetector()

    def parse_line(self, line: str) -> Optional[dict]:
        """Parse a single line of quantum-verbose output."""
        line = line.strip()
        if not line:
            return None

        sample = {}

        # Check for low entropy (greedy mode)
        match = PATTERNS['low_entropy'].search(line)
        if match:
            sample = {
                'entropy': float(match.group(1)),
                'threshold': float(match.group(2)),
                'mode': 'greedy',
                'edt_temp': None,
                'qrng_used': False,
            }
            self.samples.append(sample)
            return sample

        # Check for high entropy (EDT + QRNG)
        match = PATTERNS['high_entropy'].search(line)
        if match:
            sample = {
                'entropy': float(match.group(1)),
                'threshold': float(match.group(2)),
                'mode': 'edt_qrng',
                'edt_temp': float(match.group(3)),
                'qrng_used': True,
            }
            self.samples.append(sample)
            return sample

        # Check for basic entropy line
        match = PATTERNS['entropy_basic'].search(line)
        if match:
            sample = {
                'entropy': float(match.group(1)),
                'threshold': float(match.group(2)),
                'mode': 'unknown',
                'edt_temp': None,
                'qrng_used': None,
            }
            self.samples.append(sample)
            return sample

        # Check for QRNG mode details
        match = PATTERNS['qrng_mode'].search(line)
        if match:
            qrng_info = {
                'qrng_mode': int(match.group(1)),
                'qrng_count': int(match.group(2)),
                'rarity': match.group(3),
            }
            # Update the last sample with QRNG info
            if self.samples:
                self.samples[-1].update(qrng_info)
                # Track QRNG byte for analysis
                self.qrng_bytes.append(qrng_info['qrng_mode'] % 256)
            return qrng_info

        # Check for token
        match = PATTERNS['token'].search(line)
        if match:
            token_id = int(match.group(1))
            self.tokens.append(token_id)
            return {'token': token_id}

        return None

    def parse_stream(self, stream: TextIO):
        """Parse an entire stream of quantum-verbose output."""
        for line in stream:
            self.parse_line(line)

    def parse_file(self, filepath: str):
        """Parse a file containing quantum-verbose output."""
        with open(filepath, 'r') as f:
            self.parse_stream(f)

    def generate_demo_data(self, n_samples: int = 100):
        """Generate synthetic demo data for testing."""
        print(f"Generating {n_samples} synthetic samples...")

        for i in range(n_samples):
            entropy = random.gauss(2.5, 1.0)
            threshold = 1.5

            if entropy < threshold:
                # Low entropy -> greedy
                sample = {
                    'entropy': max(0.1, entropy),
                    'threshold': threshold,
                    'mode': 'greedy',
                    'edt_temp': None,
                    'qrng_used': False,
                }
            else:
                # High entropy -> EDT + QRNG
                edt_temp = 0.5 + (entropy - threshold) * 0.2
                qrng_mode = random.randint(0, 255)
                rarity_choices = ['common', 'uncommon', 'rare', 'very_rare']
                rarity_weights = [0.7, 0.2, 0.08, 0.02]

                sample = {
                    'entropy': entropy,
                    'threshold': threshold,
                    'mode': 'edt_qrng',
                    'edt_temp': min(1.5, edt_temp),
                    'qrng_used': True,
                    'qrng_mode': qrng_mode,
                    'qrng_count': i + 1,
                    'rarity': random.choices(rarity_choices, weights=rarity_weights)[0],
                }
                self.qrng_bytes.append(qrng_mode)

            self.samples.append(sample)

            # Generate synthetic token
            token_id = random.randint(1, 32000)
            self.tokens.append(token_id)

    def run_analysis(self) -> dict:
        """Run full analysis on captured data."""
        results = {
            'sampling_stats': self._compute_sampling_stats(),
            'entropy_stats': self._compute_entropy_stats(),
            'qrng_quality': None,
            'trajectory': None,
            'chaos': None,
        }

        # Run QRNG monitor analysis
        if self.qrng_bytes:
            for byte_val in self.qrng_bytes:
                self.qrng_monitor.update(byte_val)
            results['qrng_quality'] = self.qrng_monitor.get_summary()

        # Run trajectory analysis
        if self.tokens:
            for token_id in self.tokens:
                self.trajectory_analyzer.update(token_id)
            results['trajectory'] = {
                'classification': self.trajectory_analyzer.classify(),
                'hurst': self.trajectory_analyzer.compute_hurst(),
                'samples': len(self.tokens),
            }

        # Run chaos analysis
        if len(self.tokens) >= 20:
            entropy_values = [s['entropy'] for s in self.samples if 'entropy' in s]
            if entropy_values:
                results['chaos'] = {
                    'lyapunov': self.chaos_detector.compute_lyapunov(entropy_values),
                    'criticality': self.chaos_detector.compute_criticality_index(entropy_values),
                }

        return results

    def _compute_sampling_stats(self) -> dict:
        """Compute basic sampling statistics."""
        total = len(self.samples)
        if total == 0:
            return {'total': 0}

        greedy_count = sum(1 for s in self.samples if s.get('mode') == 'greedy')
        qrng_count = sum(1 for s in self.samples if s.get('qrng_used'))

        rarity_counts = {}
        for s in self.samples:
            rarity = s.get('rarity')
            if rarity:
                rarity_counts[rarity] = rarity_counts.get(rarity, 0) + 1

        return {
            'total': total,
            'greedy_samples': greedy_count,
            'qrng_samples': qrng_count,
            'greedy_ratio': greedy_count / total if total > 0 else 0,
            'qrng_ratio': qrng_count / total if total > 0 else 0,
            'rarity_distribution': rarity_counts,
        }

    def _compute_entropy_stats(self) -> dict:
        """Compute entropy statistics."""
        entropy_values = [s['entropy'] for s in self.samples if 'entropy' in s]
        if not entropy_values:
            return {}

        import numpy as np
        arr = np.array(entropy_values)

        edt_temps = [s['edt_temp'] for s in self.samples if s.get('edt_temp') is not None]

        stats = {
            'count': len(entropy_values),
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'median': float(np.median(arr)),
        }

        if edt_temps:
            edt_arr = np.array(edt_temps)
            stats['edt_temp_mean'] = float(np.mean(edt_arr))
            stats['edt_temp_std'] = float(np.std(edt_arr))

        return stats

    def print_report(self):
        """Print a human-readable analysis report."""
        results = self.run_analysis()

        print("\n" + "=" * 60)
        print("QRNG DATA ANALYSIS REPORT")
        print("=" * 60)

        # Sampling stats
        stats = results['sampling_stats']
        print(f"\n📊 SAMPLING STATISTICS")
        print(f"   Total samples: {stats.get('total', 0)}")
        print(f"   Greedy mode:   {stats.get('greedy_samples', 0)} ({stats.get('greedy_ratio', 0):.1%})")
        print(f"   QRNG mode:     {stats.get('qrng_samples', 0)} ({stats.get('qrng_ratio', 0):.1%})")

        if stats.get('rarity_distribution'):
            print(f"   Rarity distribution: {stats['rarity_distribution']}")

        # Entropy stats
        entropy = results['entropy_stats']
        if entropy:
            print(f"\n📈 ENTROPY STATISTICS")
            print(f"   Mean:   {entropy.get('mean', 0):.4f}")
            print(f"   Std:    {entropy.get('std', 0):.4f}")
            print(f"   Range:  [{entropy.get('min', 0):.4f}, {entropy.get('max', 0):.4f}]")
            print(f"   Median: {entropy.get('median', 0):.4f}")
            if 'edt_temp_mean' in entropy:
                print(f"   EDT temp (mean): {entropy['edt_temp_mean']:.4f}")

        # QRNG quality
        qrng = results.get('qrng_quality')
        if qrng:
            print(f"\n🎲 QRNG QUALITY METRICS")
            print(f"   Hurst exponent:  {qrng.get('hurst_exponent', 'N/A')}")
            print(f"   Min entropy:     {qrng.get('min_entropy', 'N/A')}")
            print(f"   Health:          {qrng.get('health', 'N/A')}")

        # Trajectory analysis
        traj = results.get('trajectory')
        if traj:
            print(f"\n🔄 TRAJECTORY ANALYSIS")
            print(f"   Classification: {traj.get('classification', 'N/A')}")
            print(f"   Hurst exponent: {traj.get('hurst', 'N/A'):.4f}" if traj.get('hurst') else "   Hurst exponent: N/A")
            print(f"   Samples:        {traj.get('samples', 0)}")

        # Chaos metrics
        chaos = results.get('chaos')
        if chaos:
            print(f"\n🌀 CHAOS METRICS")
            lyap = chaos.get('lyapunov')
            if lyap is not None:
                regime = "Chaotic" if lyap > 0 else "Stable" if lyap < 0 else "Neutral"
                print(f"   Lyapunov exponent: {lyap:.4f} ({regime})")
            crit = chaos.get('criticality')
            if crit is not None:
                print(f"   Criticality index: {crit:.4f}")

        print("\n" + "=" * 60)

        return results


def main():
    parser = argparse.ArgumentParser(
        description='Capture and analyze QRNG data from --quantum-verbose output',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('-f', '--file', help='Input file (default: stdin)')
    parser.add_argument('-o', '--output', help='Output JSON file for results')
    parser.add_argument('--demo', action='store_true', help='Run with synthetic demo data')
    parser.add_argument('-n', '--samples', type=int, default=100,
                        help='Number of demo samples (default: 100)')

    args = parser.parse_args()

    capture = QRNGDataCapture()

    if args.demo:
        capture.generate_demo_data(args.samples)
    elif args.file:
        print(f"Reading from file: {args.file}")
        capture.parse_file(args.file)
    else:
        print("Reading from stdin (pipe llama-cli output)...")
        print("(Use Ctrl+D to finish, or --demo for synthetic data)")
        capture.parse_stream(sys.stdin)

    if not capture.samples:
        print("No data captured. Use --demo for synthetic data or pipe quantum-verbose output.")
        return 1

    results = capture.print_report()

    if args.output:
        with open(args.output, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            def convert(obj):
                if hasattr(obj, 'item'):
                    return obj.item()
                elif hasattr(obj, 'tolist'):
                    return obj.tolist()
                return obj

            json.dump(results, f, indent=2, default=convert)
        print(f"\n📁 Results saved to: {args.output}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
