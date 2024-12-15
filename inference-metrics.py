#!/usr/bin/env python3

"""
    $ inference-metrics.py [-n 5] -- ANY PROGRAM

このスクリプトは、`--` のあとに任意のプログラムを呼び出す文字列を受け取り、そのプログラムを `-n` で指定された回数（デフォルトは 5）だけ呼び出します。
"""

import re
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

import click


def run_command(command) -> float:
    start_time = time.time()
    subprocess.run(command, check=True)
    return time.time() - start_time


@dataclass
class InferenceMetrics:
    prompt_tokens: Optional[int] = None
    ttft_ms: Optional[float] = None
    extend_tokens: Optional[int] = None
    tps: Optional[float] = None


def extract_metrics(stdout: str) -> InferenceMetrics:
    """Extract metrics from stdout"""
    metrics = InferenceMetrics()

    # Extract prompt tokens and TTFT
    prompt_match = re.search(
        r"\[Prompt\]\s*=>\s*(\d+)\s*tokens,\s*latency\s*\(TTFT\):\s*([\d.]+)\s*ms",
        stdout,
    )
    if prompt_match:
        metrics.prompt_tokens = int(prompt_match.group(1))
        metrics.ttft_ms = float(prompt_match.group(2))

    # Extract extend tokens and TPS
    extend_match = re.search(
        r"\[Extend\]\s*=>\s*(\d+)\s*tokens,\s*throughput:\s*([\d.]+)\s*tokens/s", stdout
    )
    if extend_match:
        metrics.extend_tokens = int(extend_match.group(1))
        metrics.tps = float(extend_match.group(2))

    return metrics


def calculate_average_metrics(metrics_list: list[InferenceMetrics]) -> InferenceMetrics:
    """Calculate average metrics from a list of metrics"""
    if not metrics_list:
        return InferenceMetrics()

    # First metrics for tokens (should be constant)
    result = InferenceMetrics(
        prompt_tokens=metrics_list[0].prompt_tokens,
        extend_tokens=metrics_list[0].extend_tokens,
    )

    # Average for time-based metrics
    ttft_values = [m.ttft_ms for m in metrics_list if m.ttft_ms is not None]
    tps_values = [m.tps for m in metrics_list if m.tps is not None]

    result.ttft_ms = sum(ttft_values) / len(ttft_values) if ttft_values else None
    result.tps = sum(tps_values) / len(tps_values) if tps_values else None

    return result


@click.command(
    context_settings=dict(
        ignore_unknown_options=True,
    )
)
@click.option("-n", default=5, help="Number of iterations (default: 5)")
@click.argument("command", nargs=-1, type=click.UNPROCESSED)
def main(n: int, command):
    """Run a program multiple times and measure execution time"""
    if not command:
        raise click.UsageError("No command specified after '--'")

    metrics_list = []

    for i in range(n):
        click.echo(f"\nRun {i + 1}/{n}")
        start_time = time.time()
        try:
            r = subprocess.run(command, check=True, text=True, capture_output=True)
            execution_time = time.time() - start_time
            stdout = r.stdout.strip()

            click.secho(f"stdout:\n{stdout}", fg="black")
            click.secho(f"Execution time: {execution_time:.3f}s", fg="black")

            metrics = extract_metrics(stdout)
            if any(vars(metrics).values()):  # Check if any metrics were extracted
                metrics_list.append(metrics)
                if metrics.prompt_tokens and metrics.ttft_ms:
                    click.secho(
                        f"[Prompt]  => {metrics.prompt_tokens} tokens, latency (TTFT): {metrics.ttft_ms:.2f} ms",
                        fg="blue",
                    )
                if metrics.extend_tokens and metrics.tps:
                    click.secho(
                        f"[Extend]  => {metrics.extend_tokens} tokens, throughput: {metrics.tps:.2f} tokens/s",
                        fg="blue",
                    )
        except subprocess.CalledProcessError as e:
            raise click.ClickException(f"Command failed with exit code {e.returncode}")

    if metrics_list:
        avg_metrics = calculate_average_metrics(metrics_list)
        click.echo("\nAverage Metrics:")
        if avg_metrics.prompt_tokens and avg_metrics.ttft_ms:
            click.secho(
                f"[Prompt]  => {avg_metrics.prompt_tokens} tokens, latency (TTFT): {avg_metrics.ttft_ms:.2f} ms",
                fg="green",
            )
        if avg_metrics.extend_tokens and avg_metrics.tps:
            click.secho(
                f"[Extend]  => {avg_metrics.extend_tokens} tokens, throughput: {avg_metrics.tps:.2f} tokens/s",
                fg="green",
            )


if __name__ == "__main__":
    main()
