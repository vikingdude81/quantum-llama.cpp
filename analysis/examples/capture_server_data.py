#!/usr/bin/env python3
"""
Capture Token Data from llama.cpp HTTP Server

This script queries the llama.cpp HTTP server to capture tokens and
log probabilities for analysis with the quantum-llama analysis toolkit.

Prerequisites:
    - llama.cpp server running: ./build/bin/llama-server -m model.gguf
    - requests library: pip install requests

Usage:
    # Basic usage (requires running server)
    python capture_server_data.py -p "Once upon a time"

    # With custom parameters
    python capture_server_data.py -p "Hello world" -n 50 --url http://localhost:8080

    # Demo mode (no server required)
    python capture_server_data.py --demo

    # Save results to JSON
    python capture_server_data.py -p "Test prompt" -o analysis.json
"""

import sys
import json
import argparse
import random
import math
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from trajectory_analyzer import TrajectoryAnalyzer
from chaos_detector import ChaosDetector

# Check for requests library
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class ServerDataCapture:
    """Captures and analyzes token data from llama.cpp server."""

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url.rstrip('/')
        self.tokens: List[int] = []
        self.probs: List[List[Dict[str, Any]]] = []
        self.text: str = ""

        # Analysis modules
        self.trajectory_analyzer = TrajectoryAnalyzer()
        self.chaos_detector = ChaosDetector()

    def query_server(self, prompt: str, n_predict: int = 100, n_probs: int = 10,
                     temperature: float = 0.8) -> Optional[Dict]:
        """Query the llama.cpp server for completion."""
        if not REQUESTS_AVAILABLE:
            print("Error: 'requests' library not installed.")
            print("Install with: pip install requests")
            return None

        url = f"{self.base_url}/completion"

        payload = {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": temperature,
            "n_probs": n_probs,
            "return_tokens": True,
        }

        try:
            print(f"Querying server at {url}...")
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            print(f"Error: Could not connect to server at {self.base_url}")
            print("Make sure llama-server is running:")
            print("  ./build/bin/llama-server -m model.gguf")
            return None
        except requests.exceptions.Timeout:
            print("Error: Server request timed out")
            return None
        except requests.exceptions.HTTPError as e:
            print(f"Error: HTTP {e.response.status_code}: {e.response.text}")
            return None
        except Exception as e:
            print(f"Error: {e}")
            return None

    def parse_response(self, response: Dict) -> bool:
        """Parse server response and extract tokens/probs."""
        if not response:
            return False

        # Extract generated text
        self.text = response.get('content', '')

        # Extract tokens if available
        tokens = response.get('tokens', [])
        if tokens:
            self.tokens = tokens
        else:
            # Try to get from completion_probabilities
            completion_probs = response.get('completion_probabilities', [])
            for item in completion_probs:
                if 'token' in item:
                    self.tokens.append(item['token'])

        # Extract probability distributions
        completion_probs = response.get('completion_probabilities', [])
        for item in completion_probs:
            top_probs = item.get('probs', [])
            self.probs.append(top_probs)

        return bool(self.tokens)

    def generate_demo_data(self, n_tokens: int = 100):
        """Generate synthetic demo data for testing."""
        print(f"Generating {n_tokens} synthetic tokens...")

        # Simulate token generation with varying entropy
        vocab_size = 32000

        for i in range(n_tokens):
            # Vary entropy over time (simulating different generation phases)
            phase = (i / n_tokens) * 2 * math.pi
            base_entropy = 2.0 + math.sin(phase) * 1.5

            # Generate token with entropy-based distribution
            if base_entropy < 1.5:
                # Low entropy: pick from small set
                token_id = random.choice(range(100, 200))
            else:
                # Higher entropy: broader distribution
                token_id = random.randint(100, vocab_size - 1)

            self.tokens.append(token_id)

            # Generate synthetic probability distribution
            n_probs = 10
            probs = []
            remaining = 1.0
            for j in range(n_probs):
                if j == n_probs - 1:
                    p = remaining
                else:
                    # Exponential decay
                    p = remaining * random.uniform(0.3, 0.7)
                    remaining -= p

                probs.append({
                    'tok_str': f'<tok_{token_id + j}>',
                    'prob': p,
                })

            self.probs.append(probs)

        self.text = f"[Demo: {n_tokens} synthetic tokens generated]"

    def run_analysis(self) -> Dict:
        """Run full analysis on captured data."""
        results = {
            'token_stats': self._compute_token_stats(),
            'probability_stats': self._compute_prob_stats(),
            'trajectory': None,
            'chaos': None,
            'generated_text': self.text[:500] if self.text else None,  # Truncate for report
        }

        # Run trajectory analysis
        if self.tokens:
            for token_id in self.tokens:
                self.trajectory_analyzer.update(token_id)
            results['trajectory'] = {
                'classification': self.trajectory_analyzer.classify(),
                'hurst': self.trajectory_analyzer.compute_hurst(),
                'samples': len(self.tokens),
            }

        # Run chaos analysis on log probabilities
        if self.probs and len(self.probs) >= 20:
            # Use entropy of probability distributions
            entropies = []
            for prob_dist in self.probs:
                if prob_dist:
                    entropy = 0.0
                    for p in prob_dist:
                        prob = p.get('prob', 0)
                        if prob > 0:
                            entropy -= prob * math.log2(prob)
                    entropies.append(entropy)

            if entropies:
                results['chaos'] = {
                    'lyapunov': self.chaos_detector.compute_lyapunov(entropies),
                    'criticality': self.chaos_detector.compute_criticality_index(entropies),
                    'mean_entropy': sum(entropies) / len(entropies),
                }

        return results

    def _compute_token_stats(self) -> Dict:
        """Compute basic token statistics."""
        if not self.tokens:
            return {'total': 0}

        import numpy as np
        arr = np.array(self.tokens)

        return {
            'total': len(self.tokens),
            'unique': len(set(self.tokens)),
            'mean_id': float(np.mean(arr)),
            'std_id': float(np.std(arr)),
            'min_id': int(np.min(arr)),
            'max_id': int(np.max(arr)),
        }

    def _compute_prob_stats(self) -> Dict:
        """Compute probability distribution statistics."""
        if not self.probs:
            return {}

        top1_probs = []
        entropies = []

        for prob_dist in self.probs:
            if not prob_dist:
                continue

            # Top-1 probability
            top1 = max(p.get('prob', 0) for p in prob_dist)
            top1_probs.append(top1)

            # Entropy of distribution
            entropy = 0.0
            for p in prob_dist:
                prob = p.get('prob', 0)
                if prob > 0:
                    entropy -= prob * math.log2(prob)
            entropies.append(entropy)

        if not top1_probs:
            return {}

        import numpy as np
        top1_arr = np.array(top1_probs)
        entropy_arr = np.array(entropies)

        return {
            'samples': len(top1_probs),
            'top1_prob_mean': float(np.mean(top1_arr)),
            'top1_prob_std': float(np.std(top1_arr)),
            'entropy_mean': float(np.mean(entropy_arr)),
            'entropy_std': float(np.std(entropy_arr)),
            'entropy_min': float(np.min(entropy_arr)),
            'entropy_max': float(np.max(entropy_arr)),
        }

    def print_report(self):
        """Print a human-readable analysis report."""
        results = self.run_analysis()

        print("\n" + "=" * 60)
        print("SERVER DATA ANALYSIS REPORT")
        print("=" * 60)

        # Token stats
        stats = results['token_stats']
        print(f"\n📊 TOKEN STATISTICS")
        print(f"   Total tokens:  {stats.get('total', 0)}")
        print(f"   Unique tokens: {stats.get('unique', 0)}")
        if stats.get('total', 0) > 0:
            print(f"   Token ID range: [{stats.get('min_id', 0)}, {stats.get('max_id', 0)}]")

        # Probability stats
        prob_stats = results['probability_stats']
        if prob_stats:
            print(f"\n📈 PROBABILITY STATISTICS")
            print(f"   Top-1 prob (mean): {prob_stats.get('top1_prob_mean', 0):.4f}")
            print(f"   Top-1 prob (std):  {prob_stats.get('top1_prob_std', 0):.4f}")
            print(f"   Entropy (mean):    {prob_stats.get('entropy_mean', 0):.4f}")
            print(f"   Entropy range:     [{prob_stats.get('entropy_min', 0):.4f}, {prob_stats.get('entropy_max', 0):.4f}]")

        # Trajectory analysis
        traj = results.get('trajectory')
        if traj:
            print(f"\n🔄 TRAJECTORY ANALYSIS")
            print(f"   Classification: {traj.get('classification', 'N/A')}")
            hurst = traj.get('hurst')
            if hurst is not None:
                print(f"   Hurst exponent: {hurst:.4f}")
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
            mean_ent = chaos.get('mean_entropy')
            if mean_ent is not None:
                print(f"   Mean entropy:      {mean_ent:.4f}")

        # Generated text preview
        text = results.get('generated_text')
        if text:
            print(f"\n📝 GENERATED TEXT (preview)")
            # Show first 200 chars
            preview = text[:200]
            if len(text) > 200:
                preview += "..."
            print(f"   {preview}")

        print("\n" + "=" * 60)

        return results


def main():
    parser = argparse.ArgumentParser(
        description='Capture and analyze token data from llama.cpp server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('-p', '--prompt', default="Once upon a time",
                        help='Prompt to send to server (default: "Once upon a time")')
    parser.add_argument('-n', '--tokens', type=int, default=100,
                        help='Number of tokens to generate (default: 100)')
    parser.add_argument('--url', default="http://localhost:8080",
                        help='Server URL (default: http://localhost:8080)')
    parser.add_argument('--n-probs', type=int, default=10,
                        help='Number of top probabilities to return (default: 10)')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature (default: 0.8)')
    parser.add_argument('-o', '--output', help='Output JSON file for results')
    parser.add_argument('--demo', action='store_true',
                        help='Run with synthetic demo data (no server required)')

    args = parser.parse_args()

    capture = ServerDataCapture(base_url=args.url)

    if args.demo:
        capture.generate_demo_data(args.tokens)
    else:
        if not REQUESTS_AVAILABLE:
            print("Error: 'requests' library required for server communication.")
            print("Install with: pip install requests")
            print("\nUse --demo flag to run with synthetic data instead.")
            return 1

        response = capture.query_server(
            prompt=args.prompt,
            n_predict=args.tokens,
            n_probs=args.n_probs,
            temperature=args.temperature
        )

        if not response:
            print("\nFailed to get response from server.")
            print("Use --demo flag to run with synthetic data instead.")
            return 1

        if not capture.parse_response(response):
            print("Warning: Could not extract tokens from response.")
            print("Server may not support return_tokens option.")

    if not capture.tokens:
        print("No data captured.")
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
